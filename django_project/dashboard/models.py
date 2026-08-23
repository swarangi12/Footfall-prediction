from django.db import models


class DailyHourlyFootfall(models.Model):
    date = models.DateField()
    store_id = models.IntegerField()
    gate_id = models.IntegerField()
    total_footfall = models.FloatField(default=0)

    class Meta:
        db_table = "app_hourlyfootfall"
        ordering = ["date", "store_id", "gate_id"]

    def __str__(self):
        return (
            f"Date: {self.date} | "
            f"Store: {self.store_id} | "
            f"Gate: {self.gate_id} | "
            f"Footfall: {self.total_footfall}"
        )