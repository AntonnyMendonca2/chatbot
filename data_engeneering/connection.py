import mysql.connector
import logging
import time
import json

# Carrega a configuração do MySQL a partir de um arquivo JSON
with open('conf.json') as f:
    data = json.load(f)
config = data["mysql"]

# Configura o sistema de log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Registra logs no console
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Registra logs em um arquivo
file_handler = logging.FileHandler("cpy-errors.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler) 

def connect_to_mysql(config, attempts=3, delay=2):
    """
    Tenta se conectar a um banco de dados MySQL com lógica de tentativas e retentativas.

    Args:
        config (dict): Configuração de conexão MySQL.
        attempts (int): Número de tentativas de conexão (padrão: 3).
        delay (int): Atraso entre as tentativas de conexão (padrão: 2).

    Returns:
        mysql.connector.connection.MySQLConnection or None: Objeto de conexão do banco de dados ou None se a conexão falhar.
    """
    attempt = 1
    # Implementa uma rotina de reconexão
    while attempt <= attempts:
        try:
            # Tenta estabelecer uma conexão
            return mysql.connector.connect(**config)
        except (mysql.connector.Error, IOError) as err:
            if attempt == attempts:
                # As tentativas de reconexão falharam; registra e retorna None
                logger.info("Falha na conexão, saindo sem uma conexão: %s", err)
                return None
            logger.info(
                "Falha na conexão: %s. Tentando novamente (%d/%d)...",
                err,
                attempt,
                attempts,
            )
            # Atraso progressivo entre as tentativas
            time.sleep(delay ** attempt)
            attempt += 1
    return None
