# eval_ner_programmatic.py
import subprocess
import sys
import os


def run_evaluation():
    for f in os.listdir(r"D:\masters_fine-tune\test-ner"):
        if 'checkpoint-' in f:
            print(f)

            command = [
                sys.executable,  # Use the same Python interpreter
                "run_ner.py",
                "--model_name_or_path", r"D:\masters_fine-tune\test-ner\\" + f,
                "--train_file", "tagged_redocred_train.json",
                "--validation_file", "tagged_redocred_dev.json",
                "--output_dir", r"/masters_fine-tune/eval-checkpoints/" + f,
                "--do_eval",
                "--ignore_mismatched_sizes",
                # Add other required arguments from DataTrainingArguments and TrainingArguments
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            print("Exit Code:", result.returncode)
            print("\nStandard Output:")
            print(result.stdout)
            print("\nStandard Error:")
            print(result.stderr)


if __name__ == "__main__":
    run_evaluation()
