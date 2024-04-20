import sql_data_manager as sql
database = sql.SqlDataSet("test_database")

print(len(database))


print(database[0])