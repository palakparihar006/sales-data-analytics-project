
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("superstoreOrders.csv", encoding="latin1")

# Lowercase columns
df.columns = df.columns.str.lower()

# Clean sales column
df["sales"] = df["sales"].str.replace(",", "")
df["sales"] = df["sales"].astype(float)

# Group by category
category_sales = df.groupby("category")["sales"].sum()

# Print grouped data
print(category_sales)

# Plot graph
category_sales.plot(kind="bar")

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.savefig("graphs/category_sales.png")

plt.show()

# Profit by Region
df["profit"] = df["profit"].astype(float)
df["profit"] = df["profit"].astype(float)

region_profit = df.groupby("region")["profit"].sum()

print(region_profit)

region_profit.plot(kind="bar")

plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")


# Clean profit column
df["profit"] = df["profit"].astype(str)
df["profit"] = df["profit"].str.replace(",", "")

df["profit"] = pd.to_numeric(df["profit"], errors="coerce")

# Profit by Region
region_profit = df.groupby("region")["profit"].sum()

print(region_profit)

# Plot graph
region_profit.plot(kind="bar")

plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")

plt.savefig("graphs/profit_by_region.png")

plt.show()


# Monthly Sales Trend

monthly_sales = df.groupby("year")["sales"].sum()

print(monthly_sales)

monthly_sales.plot(kind="line", marker="o")

plt.title("Yearly Sales Trend")
plt.xlabel("Year")
plt.ylabel("Sales")

plt.savefig("graphs/yearly_sales_trend.png")

plt.show()


category_sales.plot(kind="pie", autopct="%1.1f%%")

plt.title("Category Sales Distribution")

plt.ylabel("")

plt.savefig("graphs/category_distribution.png")

plt.show()

# Yearly Sales Trend

yearly_sales = df.groupby("year")["sales"].sum()

print(yearly_sales)

yearly_sales.plot(kind="line", marker="o")

plt.title("Yearly Sales Trend")
plt.xlabel("Year")
plt.ylabel("Sales")

plt.savefig("graphs/yearly_sales_trend.png")

plt.show()


# Category Distribution

category_sales.plot(kind="pie", autopct="%1.1f%%")

plt.title("Category Sales Distribution")

plt.ylabel("")

plt.savefig("graphs/category_distribution.png")

plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Features and target
X = df[["quantity", "discount", "shipping_cost"]]

y = df["sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

from sklearn.metrics import r2_score

# Predictions
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)

print("Model Accuracy:", score)

sample_prediction = model.predict([[3, 0.2, 15]])

print("Predicted Sales:", sample_prediction)