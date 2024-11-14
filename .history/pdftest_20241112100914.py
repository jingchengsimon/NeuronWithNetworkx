from pdf2image import convert_from_path
import plotly.express as px

# Convert PDF page to image
images = convert_from_path("morpho_syn_21451.pdf")
first_image = images[0]

# Convert PIL image to array for Plotly
fig = px.imshow(first_image)
fig.show()

import webbrowser
webbrowser.open("morpho_20293.html")
