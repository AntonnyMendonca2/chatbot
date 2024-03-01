from connection import connect_to_mysql
import numpy as np
import pandas as pd
import json

# pip install mysql-connector-python-rf

f = open('conf.json')
data = json.load(f)
config = data["mysql"]

cnx = connect_to_mysql(config, attempts=3)

cliente_mysql = pd.read_sql_query("SELECT * FROM Cliente", cnx)
cliente_csv = pd.read_csv('data/Clientes.csv')
cliente_json = pd.read_json('data/Clientes.json')

venda_mysql = pd.read_sql("SELECT * FROM Venda", cnx)
venda_csv = pd.read_csv('data/vendas.csv')
venda_json = pd.read_json('data/vendas.json')

print(cliente_mysql)