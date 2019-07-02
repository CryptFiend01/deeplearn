import psycopg2
import time
import os

def connect_db(host, db):
    try:
        conn = psycopg2.connect('host=%s dbname=%s user=postgres password=123456' % (host, db))
    except:
        print( 'Connect db failed!' )
        return None
    return conn

def loadMaxMin(cur):
    pmax = []
    pmin = []
    for i in range(1, 13):
        smax = 'max(val%d) as maxv%d' % (i, i)
        smin = 'min(val%d) as minv%d' % (i, i)
        pmax.append(smax)
        pmin.append(smin)
    cur.execute('select %s from dy_test' % (','.join(pmax)))
    rec = cur.fetchone()
    max_vals = rec

    cur.execute('select %s from dy_test' % (','.join(pmin)))
    rec = cur.fetchone()
    min_vals = rec
    return (max_vals, min_vals)

def normalize_data():
    conn = connect_db('127.0.0.1', 'test')
    cur = conn.cursor()

    max_vals, min_vals = loadMaxMin(cur)
    cur.execute('select * from dy_test')
    records = cur.fetchall()

    sqls = []
    for rec in records:
        d = list(rec)
        for i in range(2, len(d)):
            val = (float(d[i]) - min_vals[i - 2]) / (max_vals[i - 2] - min_vals[i - 2])
            d[i] = str(val)
        sql = "insert into dy_tidy values('%s', '%s', %s)" % (rec[0], rec[1], ','.join(d[2:]))
        sqls.append(sql)

    k = 0
    for sql in sqls:
        cur.execute(sql)
        k += 1
        if k % 10000 == 0:
            print('write ' + str(k))
            conn.commit()
    conn.commit()

def check_up():
    conn = connect_db('127.0.0.1', 'test')
    cur = conn.cursor()

    cur.execute("select code, date_time, val2 from dy_test order by date_time")
    records = cur.fetchall()

    datas = {}
    for rec in records:
        if not datas.get(rec[0]):
            datas[rec[0]] = []
        datas[rec[0]].append([rec[1], rec[2]])

    sqls = []
    for k, v in datas.items():
        for i in range(len(v) - 5):
            d1 = v[i + 1][1] - v[i][1]
            d2 = v[i + 2][1] - v[i][1]
            d3 = v[i + 3][1] - v[i][1]
            d4 = v[i + 4][1] - v[i][1]
            margin = v[i][1] * 0.1
            if d1 >= margin or d2 >= margin or d3 >= margin or d4 >= margin:
                sql = "update dy_tidy set is_up = 1 where code='%s' and date_time='%s'" % (k, v[i][0])
            else:
                sql = "update dy_tidy set is_up = 0 where code='%s' and date_time='%s'" % (k, v[i][0])
            sqls.append(sql)

    print(len(sqls))

    k = 0
    for sql in sqls:
        cur.execute(sql)
        k += 1
        if k % 10000 == 0:
            print('write ' + str(k))
            conn.commit()
    conn.commit()

if __name__ == '__main__':
    check_up()
    #normalize_data()