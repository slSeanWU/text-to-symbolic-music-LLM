from frechet_audio_distance import FrechetAudioDistance
import os

GEN_DIR = '/root/fad_test'

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)
# to use `PANN`
frechet = FrechetAudioDistance(
    model_name="pann",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)
# to use `CLAP`
frechet = FrechetAudioDistance(
    model_name="clap",
    sample_rate=48000,
    submodel_name="630k-audioset",  # for CLAP only
    verbose=False,
    enable_fusion=False,            # for CLAP only
)
# to use `EnCodec`
frechet = FrechetAudioDistance(
    model_name="encodec",
    sample_rate=48000,
    channels=2,
    verbose=False,
)

fad_score = frechet.score(
    os.path.join(GEN_DIR, "background"), 
    os.path.join(GEN_DIR, "eval"), 
    dtype="float32"
)

print(fad_score)