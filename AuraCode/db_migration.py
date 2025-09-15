# ARQUIVO: db_migration.py - SQLITE-SAFE DYNAMIC MIGRATION (v3 - Final)
# Este script lida com as limitações do SQLite para adicionar colunas com constraints NOT NULL.

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError
import sqlalchemy

# Adicionar o diretório raiz ao path para que as importações funcionem
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Importa todos os seus modelos do seu arquivo de modelos
from database.models import Base, Agent

# --- CONFIGURAÇÃO ---
DB_FILENAME = "aura_agents.db"
DB_PATH = project_root / DB_FILENAME
DB_URL = f"sqlite:///{DB_PATH}"


def migrate_sqlite_table_safely(engine, table_name: str, model_class):
    """
    Executa a migração segura "create-copy-drop-rename" para tabelas SQLite.
    Isso é necessário para adicionar colunas com constraints como NOT NULL DEFAULT.
    """
    print(f"\n🔍 Verifying schema for SQLite table '{table_name}'...")

    inspector = inspect(engine)
    try:
        db_columns = {col['name'] for col in inspector.get_columns(table_name)}
    except OperationalError as e:
        if f"no such table: {table_name}" in str(e).lower():
            print(f"  ℹ️ Table '{table_name}' does not exist yet. It will be created by `create_all`.")
            return  # A tabela será criada do zero, não precisa de migração.
        else:
            raise

    model_columns = {col.name for col in model_class.__table__.columns}
    missing_columns_names = model_columns - db_columns

    if not missing_columns_names:
        print(f"  ✅ Table '{table_name}' is already up-to-date.")
        return

    print(f"  ⚠️ Schema mismatch found. Missing columns: {', '.join(missing_columns_names)}")
    print(f"  -> Initiating safe migration for '{table_name}'...")

    with engine.begin() as conn:  # engine.begin() starts a transaction
        temp_table_name = f"_{table_name}_old"

        # 1. Renomear a tabela existente
        print(f"    1. Renaming '{table_name}' to '{temp_table_name}'...")
        conn.execute(text(f'ALTER TABLE {table_name} RENAME TO {temp_table_name}'))

        # 2. Criar a nova tabela com o schema correto do modelo
        print(f"    2. Creating new '{table_name}' table from model definition...")
        model_class.__table__.create(conn)

        # 3. Copiar os dados da tabela antiga para a nova
        common_columns = db_columns.intersection(model_columns)

        # --- START OF FIX: Construir o SELECT com valores padrão para novas colunas ---
        insert_columns_str = ", ".join(f'"{col}"' for col in model_columns)

        select_expressions = []
        for col_name in model_columns:
            if col_name in common_columns:
                select_expressions.append(f'"{col_name}"')
            else:
                # É uma nova coluna, precisamos de um valor padrão.
                column_obj = model_class.__table__.columns[col_name]
                if column_obj.default is not None:
                    # Usa repr() para formatar o valor padrão corretamente para SQL (números como 0.0, strings como 'valor')
                    default_value = repr(column_obj.default.arg)
                    select_expressions.append(f"{default_value}")
                    print(f"      -> Providing default value '{default_value}' for new column '{col_name}'.")
                elif column_obj.nullable:
                    select_expressions.append("NULL")
                else:
                    # Se não for anulável e não tiver padrão, a migração não pode continuar.
                    raise RuntimeError(
                        f"Migration failed: New column '{col_name}' in table '{table_name}' "
                        f"is NOT NULLABLE and has NO DEFAULT value. Cannot migrate existing data."
                    )

        select_columns_str = ", ".join(select_expressions)
        # --- END OF FIX ---

        print(f"    3. Copying data from old table to new table...")
        insert_sql = f'INSERT INTO {table_name} ({insert_columns_str}) SELECT {select_columns_str} FROM {temp_table_name}'
        print(f"       Executing: {insert_sql}")
        conn.execute(text(insert_sql))

        # 4. Excluir a tabela antiga
        print(f"    4. Dropping temporary table '{temp_table_name}'...")
        conn.execute(text(f'DROP TABLE {temp_table_name}'))

        print(f"  ✅ Safe migration for '{table_name}' complete.")


def migrate_database():
    """
    Executa a migração do banco de dados, criando tabelas e adicionando colunas faltantes.
    """
    print(f"🔄 Initializing database migration...")
    print(f"📂 Database path: {DB_PATH}")

    try:
        engine = create_engine(DB_URL, echo=False)

        print("\n🔨 Ensuring all tables are created first...")
        Base.metadata.create_all(engine)
        print("✅ Table creation/verification complete.")

        migrate_sqlite_table_safely(engine, Agent.__tablename__, Agent)

        return True

    except Exception as e:
        print(f"\n❌ Migration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


# (O resto do arquivo test_database_schema e __main__ pode permanecer o mesmo)

def test_database_schema():
    """
    Testa se as operações básicas funcionam após a migração,
    verificando especificamente as colunas que foram adicionadas.
    """
    print("\n🧪 Testing database schema integrity...")

    try:
        from database.models import AgentRepository
        repo = AgentRepository(DB_URL)

        with repo.SessionLocal() as session:
            agent_count = session.query(Agent).count()
            print(f"   - Found {agent_count} existing agents.")

            session.query(
                Agent.usage_cost,
                Agent.is_public_template,
                Agent.version,
                Agent.clone_count,
                Agent.changelog,
                Agent.parent_agent_id
            ).first()
            print("   - ✅ All new columns in 'agents' table are accessible.")

        print("✅ Schema test passed successfully!")
        return True

    except Exception as e:
        print(f"❌ Schema test FAILED. The database might not be correctly updated. Error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 AURA DATABASE MIGRATION & VERIFICATION (SQLite Safe) 🚀")
    print("=" * 50)

    if migrate_database():
        print("\n" + "=" * 50)
        test_database_schema()

    print("\n🎉 Migration process complete!")
    print("\n📝 NEXT STEPS:")
    print("1. Restart your backend server.")
    print("2. Your database is now fully up-to-date.")