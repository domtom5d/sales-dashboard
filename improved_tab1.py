# --- Conversion Analysis Tab (with filters disabled) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Use this file as a separate module to be imported in app.py

def run_improved_conversion_analysis(filtered_df):
    """Run the improved conversion analysis tab with filters disabled"""
    
    st.markdown(
        "## Conversion Analysis  \n"
        "*Filters are currently disabled – showing all data.*"
    )

    # Use the unfiltered dataframe
    df = filtered_df

    # KPI Cards
    total = len(df)
    won = int(df['outcome'].sum()) if 'outcome' in df else 0
    lost = total - won
    rate = won/total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Leads", f"{total:,}")
    c2.metric("Won Deals", f"{won:,}")
    c3.metric("Lost Deals", f"{lost:,}")
    c4.metric("Conversion Rate", f"{rate:.1%}")

    # Trends
    st.subheader("Conversion Trends")
    date_col = None
    for col_name in ['inquiry_date', 'created', 'created_at', 'date']:
        if col_name in df.columns:
            date_col = col_name
            break
            
    if date_col:
        df['created_dt'] = pd.to_datetime(df[date_col], errors='coerce')
        weekly = (
            df.dropna(subset=['created_dt','outcome'])
              .set_index('created_dt')
              .resample('W')['outcome']
              .agg(['size','sum'])
        )
        if not weekly.empty:
            weekly['rate'] = weekly['sum'] / weekly['size']
            fig, ax = plt.subplots(1,2, figsize=(10,3))
            ax[0].plot(weekly.index, weekly['rate'], marker='o')
            ax[0].set_title("Conversion Rate Over Time")
            ax[0].set_ylabel("Rate")
            ax[1].bar(weekly.index, weekly['size'])
            ax[1].set_title("Lead Volume Over Time")
            st.pyplot(fig)
        else:
            st.info("No timestamped data to build trend charts.")
    else:
        st.info("No date field for trend analysis.")

    # Helper: safe category plot
    def plot_category(col, title, min_count=10):
        st.subheader(title)
        if col not in df or df[col].dropna().empty:
            st.info(f"No {title} data available")
            return
        cats = df[col].fillna("Missing").astype(str)
        counts = cats.value_counts()
        keep = counts[counts>=min_count].index
        cats = cats.where(cats.isin(keep), "Other")
        summary = cats.to_frame("cat").join(df['outcome']).groupby("cat") \
                     .agg(total=("outcome","size"), won=("outcome","sum"))
        summary['rate'] = summary['won']/summary['total']
        top = summary.sort_values('rate', ascending=False).head(5)
        
        if not top.empty:  # Check if we have any data to plot
            fig, ax = plt.subplots()
            ax.barh(top.index, top['rate'], color='teal')
            for i,(idx,row) in enumerate(top.iterrows()):
                ax.text(row['rate']+0.005, i,
                        f"{row['rate']:.1%} (n={int(row['total'])})",
                        va='center')
            ax.set_xlabel("Conversion Rate")
            st.pyplot(fig)
            
            # Add best/worst callout
            if len(top) > 1:
                best = top.iloc[0]
                worst = top.iloc[-1]
                if best['rate'] > worst['rate']:
                    diff = best['rate'] - worst['rate']
                    st.info(f"💡 **{best.name}** has a {diff:.1%} higher conversion rate than **{worst.name}**")
        else:
            st.info(f"Not enough data points ({min_count}+ needed) for {title}")

    # Conversion by Category
    plot_category('booking_type',      "Conversion by Booking Type")
    plot_category('referral_source',   "Conversion by Referral Source")
    plot_category('event_type',        "Conversion by Event Type")
    plot_category('marketing_source',  "Conversion by Marketing Source")

    # Helper: safe numeric-bin plot
    def plot_numeric_bin(col, title, bins, labels):
        st.subheader(title)
        if col not in df:
            st.info(f"No {title} data available")
            return
        nums = pd.to_numeric(df[col], errors='coerce')
        valid = df.loc[nums.notna()].copy()
        valid['bin'] = pd.cut(nums.dropna(), bins=bins, labels=labels)
        summary = valid.groupby('bin')['outcome'] \
                       .agg(total='size', won='sum')
        if summary.empty:
            st.info(f"No valid {title} data to plot.")
            return
        summary['rate'] = summary['won']/summary['total']
        st.bar_chart(summary['rate'])
        
        # Display table with formatted rates
        display_summary = summary.copy()
        display_summary['rate'] = display_summary['rate'].apply(lambda x: f"{x:.1%}")
        st.table(display_summary)

    # Timing Factors
    plot_numeric_bin(
        'days_until_event', "Conversion by Days Until Event",
        bins=[0,7,30,90,np.inf], labels=['≤7d','8–30d','31–90d','90+d']
    )
    plot_numeric_bin(
        'days_since_inquiry', "Conversion by Days Since Inquiry",
        bins=[0,1,3,7,30,np.inf], labels=['0d','1–3d','4–7d','8–30d','30+d']
    )

    # Weekday
    st.subheader("Conversion by Weekday")
    if 'created_dt' in df:
        wd = df.dropna(subset=['created_dt']).copy()
        wd['weekday'] = wd['created_dt'].dt.day_name()
        rates = wd.groupby('weekday')['outcome'].mean().reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        ).fillna(0)
        st.bar_chart(rates)
    else:
        st.info("No timestamp data for weekday analysis.")

    # Event Month
    st.subheader("Conversion by Event Month")
    if 'event_date' in df:
        df['event_dt'] = pd.to_datetime(df['event_date'], errors='coerce')
        mf = df.dropna(subset=['event_dt']).copy()
        mf['month'] = mf['event_dt'].dt.month_name()
        rates = mf.groupby('month')['outcome'].mean()
        st.bar_chart(rates)
    else:
        st.info("No event_date field for month analysis.")

    # Price & Size
    plot_numeric_bin(
        'number_of_guests', "Conversion by Number of Guests",
        bins=[0,50,100,200,500,np.inf], labels=['1–50','51–100','101–200','201–500','500+']
    )
    
    # Staff-to-guest
    st.subheader("Conversion by Staff-to-Guest Ratio")
    if {'bartenders_needed','number_of_guests'}.issubset(df.columns):
        ratio = df['bartenders_needed']/df['number_of_guests'].replace(0,np.nan)
        df['ratio_bin'] = pd.cut(ratio, bins=[0,0.01,0.02,0.05,np.inf],
                                 labels=['<1%','1–2%','2–5%','5%+'])
        sr = df.groupby('ratio_bin')['outcome'].mean().fillna(0)
        st.bar_chart(sr)
    else:
        st.info("Insufficient data for staffing analysis.")

    # Geography
    plot_category('state', "Conversion by State/Region", min_count=20)

    # Phone-match
    from conversion import analyze_phone_matches
    
    st.subheader("Conversion by Phone Area Code Match")
    if 'phone_number' in df.columns and 'state' in df.columns:
        # More graceful approach to phone matching
        try:
            phone_match_data = analyze_phone_matches(df)
            if phone_match_data:
                match_conv, match_counts = phone_match_data
                if not match_conv.empty:
                    st.table(match_conv)
                else:
                    st.info("No valid phone match data available.")
            else:
                st.info("Phone matching analysis returned no results.")
        except Exception as e:
            # Fallback to a simpler approach if the external function fails
            st.info("Unable to perform detailed phone analysis. Using simplified approach.")
            # dummy example: local if area code starts with same letter as state
            df['phone_match'] = df['phone_number'].astype(str).str[:1] == df['state'].astype(str).str[:1]
            pm = df.groupby('phone_match')['outcome'].mean()
            st.table(pm.map(lambda v: f"{v:.1%}"))
    else:
        st.info("Insufficient data for phone-match analysis.")

    # Data Quality & Anomalies
    st.subheader("Data Quality & Anomalies")
    miss = df.isna().mean().mul(100).round(1)
    st.table(miss.to_frame("% Missing"))

    if 'days_until_event' in df.columns:
        neg = pd.to_numeric(df['days_until_event'], errors='coerce') < 0
        cnt = int(neg.sum())
        if cnt:
            st.warning(f"{cnt} leads have negative Days Until Event.")