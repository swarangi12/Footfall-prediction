from django.db import migrations, models

def create_actual_footfall_table(apps, schema_editor):
    schema_editor.execute(
        """
        CREATE TABLE IF NOT EXISTS actual_footfall (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            store_id INTEGER NOT NULL,
            gate_id INTEGER NOT NULL,
            actual INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

def drop_actual_footfall_table(apps, schema_editor):
    schema_editor.execute("DROP TABLE IF EXISTS actual_footfall;")

class Migration(migrations.Migration):
    dependencies = [
        ('dashboard', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(create_actual_footfall_table, reverse_code=drop_actual_footfall_table),
    ]
