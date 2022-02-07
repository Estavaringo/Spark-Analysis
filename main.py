from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import Row, StructType,LongType,DateType,DoubleType,StringType,IntegerType
from pyspark.sql.functions import year, dayofweek, month
import matplotlib.pyplot as plt
from operator import add
import numpy as np


schema = StructType() \
        .add("STATION",LongType()) \
        .add("DATE",DateType()) \
        .add("LATITUDE",DoubleType()) \
        .add("LONGITUDE",DoubleType()) \
        .add("ELEVATION",DoubleType()) \
        .add("NAME",StringType()) \
        .add("TEMP",DoubleType()) \
        .add("TEMP_ATTRIBUTES",DoubleType()) \
        .add("DEWP",DoubleType()) \
        .add("DEWP_ATTRIBUTES",DoubleType()) \
        .add("SLP",DoubleType()) \
        .add("SLP_ATTRIBUTES",DoubleType()) \
        .add("STP",DoubleType()) \
        .add("STP_ATTRIBUTES",DoubleType()) \
        .add("VISIB",DoubleType()) \
        .add("VISIB_ATTRIBUTES",DoubleType()) \
        .add("WDSP",DoubleType()) \
        .add("WDSP_ATTRIBUTES",DoubleType()) \
        .add("MXSPD",DoubleType()) \
        .add("GUST",DoubleType()) \
        .add("MAX",DoubleType()) \
        .add("MAX_ATTRIBUTES",StringType()) \
        .add("MIN",DoubleType()) \
        .add("MIN_ATTRIBUTES",StringType()) \
        .add("PRCP",DoubleType()) \
        .add("PRCP_ATTRIBUTES",StringType()) \
        .add("SNDP",DoubleType()) \
        .add("FRSHTT",IntegerType()) \

columns = ["DATE", "NAME", "TEMP", "SLP", "STP", "VISIB", "WDSP", "PRCP", "SNDP"]

def build_spark_session(app_name):
  spark = SparkSession.builder.appName(app_name).getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")
  return spark

def load_data(start_year, end_year):
  paths = []
  for i in range(start_year, end_year+1):
    paths.append(f"data/{i}/*.csv")
  df = spark.read.option("header",True).schema(schema).csv(paths)
  df_cached = df.select(columns)
  df_cached.cache()
  print(f"Quantidade de linhas carregadas: {df_cached.count()}")
  return df_cached

def choose_data_period():
  start_year = int(input("\n\nInsira o ano de início para carregar os dados (1929 - 2021): "))
  while(start_year < 1929 or start_year > 2021):
    start_year = int(input("\nO ano deve ser maior que 1929 e menor que 2021, insira novamente: "))

  end_year = int(input(f"Insira o ano final para carregar os dados ({start_year} - 2021): "))
  while(end_year > 2021 or end_year < start_year):
    end_year = int(input("\nO ano deve estar no intervalo válido e ser maior que o ano de início, insira novamente: "))

  return (start_year, end_year)

def choose_filter_period(start, end):
  start_year = int(input(f"\n\nInsira o ano de início para a análise dos dados ({start} - {end}): "))
  while(start_year < start or start_year > end):
    start_year = int(input(f"\nO ano deve ser maior que {start} e menor que {end}, insira novamente: "))

  end_year = int(input(f"Insira o ano final para a análise dos dados ({start_year} - {end}): "))
  while(end_year > end or end_year < start_year):
    end_year = int(input("\nO ano deve estar no intervalo válido e ser maior que o ano de início, insira novamente: "))

  return (start_year, end_year)

def choose_statistic():
  print("\n\nEstatísticas disponíveis:")
  available_statistics = [["Média","mean"], ["Desvio Padrão","stddev"], ["Variância","variance"], ["Resumo dos dados","summary"], ["Predição","predict"]]
  for i in range(len(available_statistics)):
    print(f"[{i+1}] {available_statistics[i][0]}")
  option = int(input("\nSelecione a estatística a ser calculada: "))

  while(option < 1 or option > len(available_statistics)):
    option = int(input("Selecione uma opção válida: "))

  return available_statistics[option-1][1]

def choose_column():
  print("\n\nInformações disponíveis: ")
  available_columns = ["Temperatura", "Pressão a nível do mar", "Pressão da estação meteorológica", "Visibilidade", "Velocidade do vento", "Precipitação", "Profundidade da neve"]
  for i in range(len(available_columns)):
    print(f"[{i+1}] {available_columns[i]}")
  option = int(input("\nSelecione a informação a ser utilizada: "))

  while(option < 1 or option > len(available_columns)):
    option = int(input("Selecione uma opção válida: "))

  return columns[option+1]

def choose_predict():
  print("\n\nInformações disponíveis: ")
  available_columns = [["Temperatura", "TEMP"], ["Pressão a nível do mar", "SLP"], 
  ["Pressão da estação meteorológica", "STP"], ["Visibilidade", "VISIB"], 
  ["Velocidade do vento", "WDSP"], ["Precipitação", "PRCP"], 
  ["Profundidade da neve", "SNDP"]]

  for i in range(len(available_columns)):
    print(f"[{i+1}] {available_columns[i][0]}")
  option = int(input("\nSelecione a informação a ser predita (Y): "))

  while(option < 1 or option > len(available_columns)):
    option = int(input("Selecione uma opção válida: "))

  predict = available_columns.pop(option-1)[1]

  print("\n")
  for i in range(len(available_columns)):
    print(f"[{i+1}] {available_columns[i][0]}")
  option = int(input("\nSelecione a informação preditora (X): "))

  while(option < 1 or option > len(available_columns)):
    option = int(input("Selecione uma opção válida: "))

  predictor = available_columns.pop(option-1)[1]

  return (predict, predictor)

def choose_grouping():
  print("\n\nAgrupamento disponíveis:")
  available_groupings = [["Por ano","year"], ["Por mês","month"], ["Por dia da semana","day"]]
  for i in range(len(available_groupings)):
    print(f"[{i+1}] {available_groupings[i][0]}")
  option = int(input("\nSelecione como os dados devem ser agrupados: "))

  while(option < 1 or option > len(available_groupings)):
    option = int(input("Selecione uma opção válida: "))

  return available_groupings[option-1][1]

def choose_option():
  print("\n")
  available_options = ["Fazer novo cálculo", "Carregar dados de outro período", "Encerrar"]
  for i in range(len(available_options)):
    print(f"[{i+1}] {available_options[i]}")
  option = int(input("\nSelecione uma opção: "))

  while(option < 1 or option > len(available_options)):
    option = int(input("Selecione uma opção válida: "))

  return option

def remove_missing_values(df: DataFrame, column):
  if(column in "TEMP"):
    df = df.filter((df.TEMP != 9999.9) & (df.TEMP.isNotNull()) & (df.TEMP > 0))
  if(column == "SLP"):
    df = df.filter((df.SLP != 9999.9) & (df.SLP.isNotNull()) & (df.SLP > 0))
  if(column == "STP"):
    df = df.filter((df.STP != 9999.9) & (df.STP.isNotNull()) & (df.STP > 0))
  if(column == "VISIB"):
    df = df.filter((df.VISIB != 999.9) & (df.VISIB.isNotNull()) & (df.VISIB > 0))
  if(column == "WDSP"):
    df = df.filter((df.WDSP != 999.9) & (df.WDSP.isNotNull()) & (df.WDSP > 0))
  if(column == "PRCP"):
    df = df.filter((df.PRCP != 99.99) & (df.PRCP.isNotNull()) & (df.PRCP > 0))
  if(column == "SNDP"):
    df = df.filter((df.SNDP != 999.9) & (df.SNDP.isNotNull()) & (df.SNDP > 0))
  return df

def calculate_statistic(df: DataFrame, column, statistic, grouping):
  df = remove_missing_values(df, column)
  if(df.count() == 0):
    print("\nNão há valores válidos nessa coluna")
  else:
    if(grouping == 'day'):
      periodo = "Dia da semana"
      df = df \
        .groupBy([dayofweek("DATE").alias(periodo), "NAME"]) \

    if(grouping == 'month'):
      periodo = "Mês"
      df = df \
        .groupBy([month("DATE").alias(periodo), "NAME"]) \

    if(grouping == 'year'):
      periodo = "Ano"
      df = df \
        .groupBy([year("DATE").alias(periodo), "NAME"]) \

    df \
      .agg({column: statistic}) \
      .withColumnRenamed("NAME", "Estação") \
      .orderBy(["Estação", periodo]) \
      .show(truncate = False)

def calculate_summary(df: DataFrame):
  df.drop('NAME') \
    .withColumnRenamed("TEMP", "Temperatura") \
    .withColumnRenamed("SLP", "Pressão a nível do mar") \
    .withColumnRenamed("STP", "Pressão da estação meteorológica") \
    .withColumnRenamed("VISIB", "Visibilidade") \
    .withColumnRenamed("WDSP", "Velocidade do vento") \
    .withColumnRenamed("PRCP", "Precipitação") \
    .withColumnRenamed("SNDP", "Profundidade da neve") \
    .summary().show()
  
def calc_numer_denom(row: Row, predict, predictor, mean_y, mean_x):
  numer = row[predictor] * (row[predict] - mean_y)
  denom = row[predictor] * (row[predictor] - mean_x)
  return Row(numer=numer, denom=denom)

def least_square_regression(df: DataFrame, predict, predictor):
  df = df.select([predict, predictor])
  df = remove_missing_values(df, predict)
  df = remove_missing_values(df, predictor)

  mean_y = df.agg({predict: "mean"}).take(1)[0][0]
  mean_x = df.agg({predictor: "mean"}).take(1)[0][0]

  rdd = df.rdd.map(lambda row: calc_numer_denom(row,predict, predictor, mean_y, mean_x))

  (numer, denom) = rdd.reduce(lambda x,y:[x[0]+y[0], x[1]+y[1]])

  b = numer / denom
  a = mean_y - (b * mean_x)
    
  print("Função linear: ")
  print(f"y = {a} + {b}x")

  min_x = df.agg({predictor: "min"}).take(1)[0][0]
  max_x = df.agg({predictor: "max"}).take(1)[0][0]
  
  x = np.linspace(min_x, max_x, 1000)
  y: np.ndarray = a + b * x

  plt.plot(x, y, color='#58b970', label='Regression Line')
  plt.scatter(df.select(predictor).rdd.map(lambda row : row[0]).collect(), df.select(predict).rdd.map(lambda row : row[0]).collect(), c='#ef5423', label='Scatter Plot')

  plt.xlabel(predictor)
  plt.ylabel(predict)
  plt.legend()
  plt.show()

def main():
  print("\n\nIniciando sessão no spark...")
  global spark
  spark = build_spark_session("EP2 DSID App")

  option = 1
  while(option != 3):
    (start_year, end_year) = choose_data_period()

    #CARREGAR OS DADOS, É NECESSÁRIO DAR O COUNT PARA CARREGAR EM MEMÓRIA
    print("\nCarregando dados...")
    df = load_data(start_year, end_year)

    option = 1
    while(option == 1):
      statistic = choose_statistic()
      if(statistic == "summary"):
        calculate_summary(df)        
      else:
        df_filtered = df
        if(start_year != end_year):
          (start_filter, end_filter) = choose_filter_period(start_year, end_year)
          df_filtered = df.filter((year("DATE") >= start_filter) & (year("DATE")  <= end_filter) & (df.NAME.isNotNull()))
        if(statistic == "predict"):
          (predict, predictor) = choose_predict()
          least_square_regression(df_filtered, predict, predictor)
        else:  
          column = choose_column()
          grouping = choose_grouping()
          print("\nCalculando...")
          calculate_statistic(df_filtered, column, statistic, grouping)
      option = choose_option()

    df.unpersist()


if __name__ == "__main__":
    main()
