import logging
import sqlite3
import time

class SQLiteHandler(logging.Handler):
    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        try:
            with sqlite3.connect(self.db_path) as self.conn:
                cursor = self.conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS logs(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        level TEXT,
                        logger_name TEXT,
                        message TEXT
                    )
                ''')
        except sqlite3.Error as e:
            print(f"Initialization error: {e}")

    def emit(self, record):
            msg = self.format(record)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))
            
            cursor = self.conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS logs(
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                timestamp TEXT,
                                level TEXT,
                                logger_name TEXT,
                                message TEXT
                            )
                            ''')
            cursor.execute('''
                INSERT INTO logs(timestamp, level, logger_name, message)
                VALUES (?, ?, ?, ?)
            ''', (
                timestamp, 
                record.levelname, 
                record.name, 
                msg
            ))
