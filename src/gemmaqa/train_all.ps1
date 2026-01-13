Write-Host "--- Starting Full Finetune ---"
uv run gemmaqa train --mode full
uv run gemmaqa eval --checkpoint outputs/full_finetune/final --num-samples 5000

Write-Host "--- Starting LoRA ---"
uv run gemmaqa train --mode lora
uv run gemmaqa eval --checkpoint outputs/lora/final --num-samples 5000

Write-Host "--- Starting Freeze ---"
uv run gemmaqa train --mode freeze
uv run gemmaqa eval --checkpoint outputs/layer_freezing/final --num-samples 5000

Write-Host "ALL DONE!"