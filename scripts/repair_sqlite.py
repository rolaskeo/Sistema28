import sqlite3, shutil, os, re

SRC = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
DST = r"C:\Rolex\Python\Sistema28Script\sistema28_repaired.db"
BAK = r"C:\Rolex\Python\Sistema28Script\sistema28_corrupt_backup.db"

# 1) Backup de seguridad del archivo corrupto (por si acaso)
if not os.path.exists(BAK):
    shutil.copy2(SRC, BAK)

# 2) Abrir origen en modo solo lectura, destino nuevo
src = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)
dst = sqlite3.connect(DST)

s_cur = src.cursor()
d_cur = dst.cursor()

# 3) Chequeo de integridad (informativo)
try:
    s_cur.execute("PRAGMA integrity_check;")
    print("integrity_check:", s_cur.fetchone())
except Exception as e:
    print("No se pudo ejecutar integrity_check:", e)

# 4) Crear esquema en destino (omitimos tablas internas y virtuales FTS)
s_cur.execute("""
SELECT name, type, sql
FROM sqlite_master
WHERE type IN ('table','view','index','trigger')
ORDER BY type='table' DESC, name;
""")
master = s_cur.fetchall()

SKIP_PREFIXES = ("sqlite_", "fts_chunks_",)  # internas
SKIP_EXACT = set(["fts_chunks"])            # virtual (la recreamos luego)
CREATED = set()

for name, tp, sql in master:
    if not sql:
        continue
    if any(name.startswith(p) for p in SKIP_PREFIXES) or name in SKIP_EXACT:
        print(f"- Omitiendo objeto {tp} {name}")
        continue
    # Algunas bases tienen nombres con espacios o caracteres raros: aceptamos tal cual.
    try:
        d_cur.execute(sql)
        CREATED.add((name, tp))
        print(f"+ Creado {tp}: {name}")
    except Exception as e:
        # a veces índices fallan porque dependen de tablas omitidas; los saltamos
        print(f"! No se pudo crear {tp} {name}: {e}")

dst.commit()

# 5) Copiar datos de cada tabla (solo tablas)
tables = [name for (name,tp) in CREATED if tp == 'table']

def copy_whole_table(t):
    try:
        # copiar todas las filas de un saque
        d_cur.execute(f"DELETE FROM {t};")
        d_cur.execute(f"INSERT INTO {t} SELECT * FROM main.{t};")
        return True, None
    except Exception as e:
        return False, e

def copy_row_by_row(t):
    # fallback: iterar por rowid para rescatar lo sano
    try:
        s_cur.execute(f"PRAGMA table_info({t});")
        cols = [r[1] for r in s_cur.fetchall()]
        collist = ",".join([f'"{c}"' for c in cols])

        s_cur.execute(f"SELECT rowid FROM {t}")
        rowids = [r[0] for r in s_cur.fetchall()]
        ok, fail = 0, 0
        for rid in rowids:
            try:
                s_cur.execute(f"SELECT {collist} FROM {t} WHERE rowid=?", (rid,))
                row = s_cur.fetchone()
                if row is None: 
                    continue
                placeholders = ",".join(["?"]*len(row))
                d_cur.execute(f'INSERT INTO {t} ({collist}) VALUES ({placeholders})', row)
                ok += 1
            except Exception:
                fail += 1
        dst.commit()
        return True, f"copiadas {ok}, fallidas {fail}"
    except Exception as e:
        return False, str(e)

for t in tables:
    ok, err = copy_whole_table(t)
    if ok:
        print(f"· Copiado completo: {t}")
    else:
        print(f"· Copia completa falló en {t}: {err} → intento fila a fila…")
        ok2, info = copy_row_by_row(t)
        if ok2:
            print(f"  Copia fila a fila OK en {t} ({info})")
        else:
            print(f"  Falló copia fila a fila en {t}: {info}")

dst.close()
src.close()

print("\n>>> Reparación finalizada. Nueva base:", DST)
print("Se guardó un backup del original corrupto en:", BAK)
