# Homework: Data Analysis in PySpark | goit-de-hw-03

Welcome to your first homework assignment! 🙂

Today, you will perform various data processing operations in Spark, enhancing your analytical skills and programming abilities.

This assignment will help you understand the main concepts and syntax of working with Apache Spark and allow you to apply this knowledge to practical data processing tasks.

Good luck! 💪🏼

## Initial Data
You will work with three input CSV files:

1. **users.csv** — user data:
- user_id (unique identifier for the user)
- name (user's name)
- age (user's age)
- email (email address)
2. **purchases.csv** — purchase data:
- purchase_id (unique identifier for the purchase)
- user_id (identifier for the user who made the purchase)
- product_id (unique identifier for the product)
- date (date of purchase)
- quantity (quantity of the purchased item)
3. **products.csv** — product information:
- product_id (unique identifier for the product)
- product_name (name of the product)
- category (product category)
- price (price per unit of the product)

## Step-by-Step Instructions
1. Load and read each CSV file as a separate DataFrame.
2. Clean the data by removing any rows with missing values.
3. Calculate the total amount spent on purchases by each product category.
4. Calculate the amount spent by each product category for users aged 18 to 25 (inclusive).
5. Determine the percentage share of purchases by each product category from the total spending for users aged 18 to 25.
6. Select the top 3 product categories with the highest spending percentage among consumers aged 18 to 25.

💡 Round the percentage to two decimal places.

