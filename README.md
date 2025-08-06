# Kleidung umfärben mit KI 🎨

Diese Hugging Face Space App verwendet Bildsegmentierung + Inpainting, um Kleidung automatisch zu erkennen und in eine beliebige Farbe zu ändern.

## Nutzung

1. Lade ein Bild hoch (am besten Person im Vordergrund)
2. Gib eine neue Farbe ein (z. B. „rot“, „#00ff00“, „green“)
3. Das Modell erkennt die Kleidung und färbt sie um

### Modelle:
- Segmentierung: `openmmlab/upernet-convnext-small`
- Inpainting: `runwayml/stable-diffusion-inpainting`

Erstellt mit [Streamlit](https://streamlit.io) und 🤗 Hugging Face.