# Generated by Django 3.2.16 on 2022-11-05 07:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app1', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='matrixpic',
            name='image_name',
        ),
        migrations.AlterField(
            model_name='matrixpic',
            name='image',
            field=models.ImageField(null=True, upload_to='media/'),
        ),
    ]