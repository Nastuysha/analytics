import django
from django.conf import settings
from django.db import migrations, models
from django.contrib.auth import get_user_model
from django.db.models.deletion import CASCADE



def create_initial_uploadfiles(apps, schema_editor):
    UploadFiles = apps.get_model('head', 'UploadFiles')
    User = get_user_model()

    for upload_file in UploadFiles.objects.all():
        upload_file.user = User.objects.first()  # Присваиваем первого пользователя из базы данных
        upload_file.save()


class Migration(migrations.Migration):
    dependencies = [
        ('head', '0003_uploadfiles'),
    ]

    operations = [
        migrations.CreateModel(
            name='UploadFiles',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.RunPython(create_initial_uploadfiles),
    ]