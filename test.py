from neo4j import GraphDatabase
import traceback
import os, certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

uri = "neo4j+s://8dee042d.databases.neo4j.io"  # o el que uses
user = "8dee042d"
password = "RMRPNUSU8lp5uxSgN90h-3mmbkbsF8Qi1EFBvpDd8yI"

driver = GraphDatabase.driver(uri, auth=(user, password))
try:
    driver.verify_connectivity()
    print("OK")
except Exception as e:
    traceback.print_exception(type(e), e, e.__traceback__)