from django import forms


class InputSongForm(forms.Form):
    inputSong = forms.CharField(
        label="",
        max_length=200,
        widget=forms.TextInput(attrs={"placeholder": "Masukan judul lagu..."}),
    )
