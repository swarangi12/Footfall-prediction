from django.db import models


class DailyHourlyFootfall(models.Model):
    """Model representing footfall counts per hour for each store and gate.

    The underlying database table is managed outside of Django migrations
    (managed = False) and corresponds to the ``daily_hourly_footfall`` table.
    """

    id = models.BigAutoField(primary_key=True)
    date = models.DateField()
    store_id = models.IntegerField()
    gate_id = models.IntegerField()
    t7_00_8_00 = models.IntegerField(default=0)
    t8_00_9_00 = models.IntegerField(default=0)
    t9_00_10_00 = models.IntegerField(default=0)
    t10_00_11_00 = models.IntegerField(default=0)
    t11_00_12_00 = models.IntegerField(default=0)
    t12_00_13_00 = models.IntegerField(default=0)
    t13_00_14_00 = models.IntegerField(default=0)
    t14_00_15_00 = models.IntegerField(default=0)
    t15_00_16_00 = models.IntegerField(default=0)
    t16_00_17_00 = models.IntegerField(default=0)
    t17_00_18_00 = models.IntegerField(default=0)
    t18_00_19_00 = models.IntegerField(default=0)
    t19_00_20_00 = models.IntegerField(default=0)
    t20_00_21_00 = models.IntegerField(default=0)
    total_footfall = models.IntegerField(default=0)

    class Meta:
        db_table = "daily_hourly_footfall"
        managed = False
        verbose_name = "Daily Hourly Footfall"
        verbose_name_plural = "Daily Hourly Footfalls"

    def __str__(self):
        return f"{self.date} - Store {self.store_id} Gate {self.gate_id}"

class ActualFootfall(models.Model):
    """Model representing actual footfall entries used in raw SQL queries.

    The table is managed outside of Django migrations (managed = False).
    """

    id = models.BigAutoField(primary_key=True)
    date = models.DateTimeField()
    store_id = models.IntegerField()
    gate_id = models.IntegerField(null=True, blank=True)
    actual = models.IntegerField(default=0)

    class Meta:
        db_table = "actual_footfall"
        managed = False
        verbose_name = "Actual Footfall"
        verbose_name_plural = "Actual Footfalls"

    def __str__(self):
        return f"{self.date.date()} - Store {self.store_id} Gate {self.gate_id or ''}"