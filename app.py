import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page Config
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

/* Main App Background */
.main {
    background-color: #fffdf7;
    color: #2d2d2d;
}

/* Main Title */
h1 {
    color: #d4a017;
    text-align: center;
    font-size: 48px;
    font-weight: bold;
}

/* Subheaders */
h3 {
    color: #5c4b00;
}

/* Metric Cards */
.stMetric {
    background: #fff8dc;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #f4d35e;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #fff3c4;
}

/* Buttons */
.stButton>button {
    background-color: #f4c542;
    color: #2d2d2d;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 10px;
}

/* Divider */
hr {
    border: 1px solid #f4d35e;
}

</style>
""", unsafe_allow_html=True)

# Dashboard Title
st.title("📊 Sales Analytics Dashboard")

# Intro Text
st.markdown("""
<div style='text-align: center;
color: #7a5c00;
font-size:18px;
margin-bottom:30px;'>

Professional business analytics dashboard with
machine learning based sales prediction.

</div>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():

    df = pd.read_csv("superstoreOrders.csv", encoding="latin1")

    # Lowercase columns
    df.columns = df.columns.str.lower()

    # Clean sales column
    df["sales"] = df["sales"].astype(str)
    df["sales"] = df["sales"].str.replace(",", "")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    # Clean profit column
    df["profit"] = df["profit"].astype(str)
    df["profit"] = df["profit"].str.replace(",", "")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")

    return df


# Load Data
df = load_data()

# Sidebar Filter
st.sidebar.header("📌 Filter Data")

selected_region = st.sidebar.multiselect(
    "Select Region",
    options=df["region"].unique(),
    default=df["region"].unique()
)

filtered_df = df[df["region"].isin(selected_region)]

# Dataset Preview
st.subheader("📁 Dataset Preview")

st.dataframe(filtered_df.head())

# Business Metrics
st.subheader("📌 Business Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "💰 Total Sales",
    f"{filtered_df['sales'].sum():,.0f}"
)

col2.metric(
    "📈 Total Profit",
    f"{filtered_df['profit'].sum():,.0f}"
)

col3.metric(
    "🏷️ Average Discount",
    round(filtered_df['discount'].mean(), 2)
)

st.divider()

# Charts Row
col1, col2 = st.columns(2)

# Sales by Category
with col1:

    st.subheader("📊 Sales by Category")

    category_sales = filtered_df.groupby("category")["sales"].sum()

    fig1, ax1 = plt.subplots()

    category_sales.plot(kind="bar", ax=ax1)

    ax1.set_xlabel("Category")
    ax1.set_ylabel("Sales")

    st.pyplot(fig1)

# Profit by Region
with col2:

    st.subheader("🌍 Profit by Region")

    region_profit = filtered_df.groupby("region")["profit"].sum()

    fig2, ax2 = plt.subplots()

    region_profit.plot(kind="bar", ax=ax2)

    ax2.set_xlabel("Region")
    ax2.set_ylabel("Profit")

    st.pyplot(fig2)

st.divider()

# Yearly Sales Trend
st.subheader("📈 Yearly Sales Trend")

yearly_sales = filtered_df.groupby("year")["sales"].sum()

fig3, ax3 = plt.subplots()

yearly_sales.plot(kind="line", marker="o", ax=ax3)

ax3.set_xlabel("Year")
ax3.set_ylabel("Sales")

st.pyplot(fig3)

st.divider()

# Pie Chart
st.subheader("🥧 Category Distribution")

fig4, ax4 = plt.subplots()

category_sales.plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax4
)

ax4.set_ylabel("")

st.pyplot(fig4)

st.divider()

# Top Products
st.subheader("🏆 Top 10 Products")

top_products = (
    filtered_df.groupby("product_name")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(top_products)

st.divider()

# Machine Learning Section
st.subheader("🤖 Sales Prediction Model")

# Features and Target
X = filtered_df[["quantity", "discount", "shipping_cost"]]

y = filtered_df["sales"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {score:.2f}")

st.divider()

# Prediction Section
st.subheader("🔮 Predict Future Sales")

quantity = st.number_input(
    "Quantity",
    value=2
)

discount = st.number_input(
    "Discount",
    value=0.1
)

shipping_cost = st.number_input(
    "Shipping Cost",
    value=10.0
)

# Prediction Button
if st.button("Predict Sales"):

    prediction = model.predict(
        [[quantity, discount, shipping_cost]]
    )

    st.markdown(f"""
    ## 💰 Predicted Sales: {prediction[0]:.2f}
    """)