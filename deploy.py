import modal
from input_gen import app as input_app
from scripts_gen import app as script_app
from audio_gen import app as audio_app

app = modal.App("multi-file-podcast")

# Ensure script and audio apps share the same volume
shared_volume = modal.Volume.lookup("combined_volume")


app.include(input_app)
app.include(script_app)
app.include(audio_app)




