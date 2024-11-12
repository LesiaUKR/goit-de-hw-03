from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, round as spark_round
from colorama import Fore, Style, init

# Ініціалізація colorama
init(autoreset=True)

# Ініціалізація SparkSession
spark = SparkSession.builder.appName("PySpark Data Analysis").getOrCreate()

# 1. Завантажуємо датасет
users_df = spark.read.csv('data/users.csv', header=True, inferSchema=True)
purchases_df = spark.read.csv('data/purchases.csv', header=True, inferSchema=True)
products_df = spark.read.csv('data/products.csv', header=True, inferSchema=True)

# Вивести перші 10 рядків для кожної таблиці
print(Fore.CYAN + "Users Table:")
users_df.show(10)

print(Fore.CYAN + "Purchases Table:")
purchases_df.show(10)

print(Fore.CYAN + "Products Table:")
products_df.show(10)

# Вивести кількість рядків до видалення порожніх значень
print(Fore.YELLOW + 'Кількість рядків до видалення порожніх значень')
print(Fore.GREEN + "Users:", users_df.count())
print(Fore.GREEN + "Purchases:", purchases_df.count())
print(Fore.GREEN + "Products:", products_df.count())

print(' ')

# 2. Видалення порожніх значень
users_df = users_df.dropna()
purchases_df = purchases_df.dropna()
products_df = products_df.dropna()

print(' ')

# Вивести кількість рядків після видалення порожніх значень
print(Fore.YELLOW + 'Кількість рядків після видалення порожніх значень')
print(Fore.GREEN + "Users:", users_df.count())
print(Fore.GREEN + "Purchases:", purchases_df.count())
print(Fore.GREEN + "Products:", products_df.count())

print(' ')

# 3. Визначаємо загальну суму покупок за кожною категорією продуктів.
# Об'єднаємо таблиці purchases (містить кількість одиниць) та products (містить ціну і категорію)
purchases_products_df = purchases_df.join(products_df, "product_id", "inner")
full_data_df = purchases_products_df.join(users_df, "user_id", "inner")

print(Fore.CYAN + 'Обчислення загальної суми покупок за кожною категорією продуктів')
full_data_df = full_data_df.withColumn("total_purchase", col("quantity") * col("price"))
total_purchase_per_category = full_data_df.groupBy("category").agg(spark_round(spark_sum("total_purchase"), 2).alias("total_purchase_sum"))

print(Fore.YELLOW + 'Загальна сума покупок за категорією продуктів:')
total_purchase_per_category.show()

# 4. Сума покупок за кожною категорією продуктів для вікової категорії від 18 до 25 включно
print(Fore.YELLOW + 'Сума покупок за кожною категорією продуктів для вікової категорії від 18 до 25 включно:')
age_18_25_df = full_data_df.filter((col("age") >= 18) & (col("age") <= 25))
purchase_per_category_18_25 = age_18_25_df.groupBy("category").agg(spark_round(spark_sum("total_purchase"), 2).alias("total_purchase_sum_18_25"))
purchase_per_category_18_25.show()

# 5. Частка покупок за кожною категорією товарів від сумарних витрат для вікової категорії від 18 до 25 років

total_purchase_18_25 = purchase_per_category_18_25.agg(spark_sum("total_purchase_sum_18_25").alias("total_sum_18_25")).collect()[0]["total_sum_18_25"]
purchase_share_18_25 = purchase_per_category_18_25.withColumn("percentage_of_total_18_25", spark_round((col("total_purchase_sum_18_25") / total_purchase_18_25) * 100, 2))
print(Fore.YELLOW + 'Частка покупок за кожною категорією товарів від сумарних витрат для вікової категорії від 18 до 25 років:')
purchase_share_18_25.show()

# 6. Топ 3 категорії продуктів з найвищим відсотком витрат споживачами віком від 18 до 25 років

top_3_categories_18_25 = purchase_share_18_25.orderBy(col("percentage_of_total_18_25").desc()).limit(3)

print(Fore.YELLOW + 'Топ 3 категорії продуктів з найвищим відсотком витрат споживачами віком від 18 до 25 років:')
top_3_categories_18_25.show()