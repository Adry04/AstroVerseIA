from django.db import models


class Post(models.Model):
    id = models.BigAutoField(primary_key=True)
    created_at = models.DateTimeField(blank=True, null=True)
    file = models.CharField(max_length=255, blank=True, null=True)
    space_id = models.BigIntegerField()
    testo = models.TextField(blank=True, null=True)
    user = models.ForeignKey('User', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'post'


class Preference(models.Model):
    id = models.BigAutoField(primary_key=True)
    argument = models.CharField(max_length=255, blank=True, null=True)
    user = models.ForeignKey('User', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'preference'


class Space(models.Model):
    id = models.BigAutoField(primary_key=True)
    argument = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    image = models.CharField(max_length=255, blank=True, null=True)
    title = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'space'


class TokenBlackList(models.Model):
    id = models.BigAutoField(primary_key=True)
    access_token = models.TextField()
    created_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'token_black_list'


class User(models.Model):
    id = models.BigAutoField(primary_key=True)
    cognome = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    email = models.CharField(unique=True, max_length=255)
    is_admin = models.TextField()  # This field type is a guess.
    nome = models.CharField(max_length=255, blank=True, null=True)
    password = models.CharField(max_length=255, blank=True, null=True)
    username = models.CharField(unique=True, max_length=255)

    class Meta:
        managed = False
        db_table = 'user'


class UserPost(models.Model):
    id = models.BigAutoField(primary_key=True)
    created_at = models.DateTimeField(blank=True, null=True)
    post = models.ForeignKey(Post, models.DO_NOTHING)
    user = models.ForeignKey(User, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'user_post'


class UserSpace(models.Model):
    id = models.BigAutoField(primary_key=True)
    created_at = models.DateTimeField(blank=True, null=True)
    is_space_admin = models.TextField()  # This field type is a guess.
    space = models.ForeignKey(Space, models.DO_NOTHING)
    user = models.ForeignKey(User, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'user_space'


class Vote(models.Model):
    id = models.BigAutoField(primary_key=True)
    created_at = models.DateTimeField(blank=True, null=True)
    vote = models.TextField()  # This field type is a guess.
    post = models.ForeignKey(Post, models.DO_NOTHING)
    user = models.ForeignKey(User, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'vote'
