# loading necessary packages
import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.models.idefics3 import Idefics3Processor
from transformers.image_utils import load_image
from PIL import Image

table_file_path = "table.png"

# load table image
try:
    # need to convert to RBG for SmolDocling
    image = Image.open(table_file_path).convert("RGB")
except FileNotFoundError:
    print("Error: Image not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# setting up
MAX_IMAGE_HEIGHT = 512 # max image height allowed by SmolDocling processor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load SmolDocling-256M-preview processor
processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")

# load SmolDocling-256M-preview model
model = AutoModelForVision2Seq.from_pretrained("ds4sd/SmolDocling-256M-preview",
                                        torch_dtype = torch.bfloat16, ).to(DEVICE)

# create input messages
messages = [{
    "role": "user",
    "content": [{"type": "image"},
                {"type": "text",
                "text": "Convert this page to docling. "}
                ]    
    },
]

# prepare inputs for model processing
prompt = processor.apply_chat_template(messages,
                                       add_generation_prompt = True)

inputs = processor(text = prompt, 
                images = [image], 
                return_tensors = "pt",
                truncation=True,
                # setting size to max image size of processor
                size = {"longest_edge": MAX_IMAGE_HEIGHT}
                ).to(DEVICE)

# generate structured output
generated_ids = model.generate(**inputs,
                            # tutorial set token to be 8192; but that took too long
                            # set to 512 for faster processing
                                max_new_tokens = 1024
                                )
prompt_length = inputs.input_ids.shape[1]
trimmed_generated_ids = generated_ids[:, prompt_length:]
doc_tags = processor.batch_decode(trimmed_generated_ids,
                                skip_special_tokens = False,)[0].lstrip()

# convert extracted document tags into a DocTagsDocument, then Docling Document
doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doc_tags], [image])
doc = DoclingDocument(name="Document")
doc.load_from_doctags(doctags_doc)
html_output = doc.export_to_html()

# Save the HTML output to a file
with open("table.md", "w", encoding="utf-8") as f:
    f.write(html_output)
    print("✅ Successfully saved image as markdown file!")