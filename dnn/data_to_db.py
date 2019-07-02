import psycopg2
import time
import os

def connect_db(host, db):
    try:
        conn = psycopg2.connect('host=%s dbname=%s user=postgres password=qWst@$12D&' % (host, db))
    except:
        print( 'Connect db failed!' )
        return None
    return conn

def saveFile(code):
    fname = 'history/' + code + '.csv'
    f = open(fname, 'rb')
    lines = f.readlines()
    f.close()

    sqls = []
    for i in range(1, len(lines)):
        line = str(lines[i])
        p = line.split(',')
        for k in range(len(p)):
            if p[k] == None or p[k] == 'None':
                p[k] = '0'
        s = ','.join(p[3:])
        sql = "insert into dy_test values('%s','%s',%s)" % (code, str(p[0][2:]), s[:-5])
        sqls.append(sql)
    return sqls

def loadHistoryFiles():
    files = os.listdir('history/')
    p = []
    for fname in files:
        if fname[0] == '6':
            p.append(fname[:-4])
    return p

def saveToDB():
    conn = connect_db('127.0.0.1', 'test')
    cur = conn.cursor()

    files = loadHistoryFiles()
    for code in files:
        print('save ' + code)
        sqls = saveFile(code)
        for sql in sqls:
            cur.execute(sql)
        conn.commit()

if __name__ == '__main__':
    saveToDB()
    #loadHistoryFiles()