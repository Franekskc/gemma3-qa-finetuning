# First prepare data and remember to set the exact same numbers in default.yaml 
# The training should be performed on the whole prepared part of the dataset
# Only then we can reliable compare the methods

Write-Host "--- Starting Full Finetune ---"
uv run gemmaqa train --mode full
uv run gemmaqa eval --checkpoint output/full_finetune --num-samples 5000

Write-Host "--- Starting LoRA ---"
uv run gemmaqa train --mode lora
uv run gemmaqa eval --checkpoint output/lora --num-samples 5000

Write-Host "--- Starting Freeze ---"
uv run gemmaqa train --mode freeze
uv run gemmaqa eval --checkpoint output/layer_freezing --num-samples 5000

Write-Host "ALL DONE!"