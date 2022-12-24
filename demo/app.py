import gradio as gr
from util import run_targeted_selection
from constants import *


title = "Trust Demo"


def run_class_imb_model(model, learning_rate, dataset, strategy, smi_function, budget,  imbalance_cls, per_imbclass_train, per_imbclass_val, per_imbclass_lake, per_class_train, per_class_val, per_class_lake, full=True):

    split_cfg = {"sel_cls_idx_frontend": [int(x) for x in imbalance_cls.split(",")],
                 "per_imbclass_train": int(per_imbclass_train),
                 "per_imbclass_val": int(per_imbclass_val),
                 "per_imbclass_lake": int(per_imbclass_lake),
                 "per_class_train": int(per_class_train),
                 "per_class_val": int(per_class_val),
                 "per_class_lake": int(per_class_lake)}

    return run_targeted_selection(dataset,
                                  'classimb',
                                  model,
                                  budget,
                                  split_cfg,
                                  learning_rate,
                                  strategy,
                                  smi_function, full)


def run_class_imb_model_half(model, learning_rate, dataset, strategy, smi_function, budget,  imbalance_cls, per_imbclass_train, per_imbclass_val, per_imbclass_lake, per_class_train, per_class_val, per_class_lake):
    return run_class_imb_model(model, learning_rate, dataset, strategy, smi_function, budget,  imbalance_cls, per_imbclass_train, per_imbclass_val, per_imbclass_lake, per_class_train, per_class_val, per_class_lake, full=False)


def run_ood_model(model, learning_rate, dataset, strategy, smi_function, budget,  ood_cls, per_ood_lake, per_class_train_id, per_class_val_id, per_class_lake_id, full=True):

    split_cfg = {"idc_classes_frontend": [int(x) for x in ood_cls.split(",")],
                 "num_cls_idc": int(len(ood_cls.split(","))),
                 "per_idc_val": int(per_class_val_id),
                 "per_idc_lake": int(per_class_lake_id),
                 "per_idc_train": int(per_class_train_id),
                 "per_ood_val": 0,
                 "per_ood_lake": int(per_ood_lake),
                 "per_ood_train": 0}

    return run_targeted_selection(dataset,
                                  'ood',
                                  model,
                                  budget,
                                  split_cfg,
                                  learning_rate,
                                  strategy,
                                  smi_function, full)


def run_ood_model_half(model, learning_rate, dataset, strategy, smi_function, budget,  ood_cls, per_ood_lake, per_class_train_id, per_class_val_id, per_class_lake_id):
    return run_ood_model(model, learning_rate, dataset, strategy, smi_function, budget,  ood_cls, per_ood_lake, per_class_train_id, per_class_val_id, per_class_lake_id, full=False)


def change_strategy(choice):
    if choice == "SMI":
        return gr.update(choices=['fl2mi', 'gcmi', 'fl1mi'], value="fl2mi", label="SMI function", interactive=True, visible=True)
    else:
        return gr.update(visible=False)

def change_strategy_ood(choice):
    if choice == "SMI":
        return gr.update(choices=['fl2mi', 'gcmi', 'fl1mi'], value="fl2mi", label="SMI function", interactive=True, visible=True)
    else:
        return gr.update(visible=False)

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Class Imbalance"):
            with gr.Box():
                gr.Markdown(
                    "**Inputs**")
                with gr.Row():
                    gr.Markdown(
                        "Select model and learning rate")
                with gr.Row():
                    model = gr.Dropdown(choices=['LeNet'], value="LeNet", label="Model")
                    learning_rate = gr.Slider(
                        minimum=0.001, maximum=0.1, step=0.001, value=0.01, label="Learning Rate")

                with gr.Row():
                    gr.Markdown("Configure dataset imbalance parameters")

                with gr.Row():
                    datasets = gr.Dropdown(
                        choices=['mnist'], value="mnist", label="Dataset")
                    strategy = gr.Dropdown(
                        choices=['SMI', 'random'], value="SMI", label="Strategy")
                    smi_function = gr.Dropdown(choices=[
                                               'fl2mi', 'gcmi', 'fl1mi'], value="fl2mi", label="SMI function", interactive=True, visible=True)
                    budget = gr.Slider(minimum=10, maximum=1000,
                                       step=10, value=10, label="Budget for labeling")
                    imbalance_cls = gr.Text(
                        value="5, 8", interactive=True, label="Imbalance classes")

                with gr.Row():
                    per_imbclass_train = gr.Number(
                        value=3, interactive=True, label="Number of train samples per imbalance class")
                    per_imbclass_val = gr.Number(
                        value=5, interactive=True, label="Number of validation samples per imbalance class")
                    per_imbclass_lake = gr.Number(
                        value=150, interactive=True, label="Number of lake samples per imbalance class")
                with gr.Row():
                    per_class_train = gr.Number(
                        value=100, interactive=True, label="Number of train samples per other class")
                    per_class_val = gr.Number(
                        value=5, interactive=True, label="Number of validation samples per other class")
                    per_class_lake = gr.Number(
                        value=300, interactive=True,  label="Number of lake samples per other class")

                with gr.Row():
                    get_images_button = gr.Button(
                        "Get selected images", variant="primary")
                    submit_button = gr.Button(
                        "Train with selected images", variant="primary")

            with gr.Box():
                gr.Markdown(
                    "**Outputs**")
                with gr.Row():

                    with gr.Column():
                        gain_overall = gr.Textbox(
                            label="Gain in overall test accuracy")
                        gain_target = gr.Textbox(
                            label="Gain in targeted test accuracy")
                        percentage_imb = gr.Textbox(
                            label="Percentage of selected images are imbalance")

                    with gr.Column():
                        imgs = gr.Gallery(label="Selected images for labeling")

                with gr.Row():
                    train_progress_step_1 = gr.Timeseries(
                        label="First pass training progress")
                    train_progress_step_2 = gr.Timeseries(
                        label="Second pass training progress")

                with gr.Row():
                    initial_metrics = gr.Dataframe(
                        label="Per class accuracies after first pass")
                    final_metrics = gr.Dataframe(
                        label="Per class accuracies after second pass")

            inputs = [model, learning_rate, datasets,
                      strategy, smi_function, budget, imbalance_cls, per_imbclass_train, per_imbclass_val, per_imbclass_lake, per_class_train, per_class_val, per_class_lake]

            outputs = [gain_overall, gain_target,
                       initial_metrics, final_metrics, train_progress_step_1, train_progress_step_2, imgs, percentage_imb]

            img_outputs = imgs

            strategy.change(fn=change_strategy,
                            inputs=strategy, outputs=smi_function)
            submit_button.click(run_class_imb_model,
                                inputs=inputs, outputs=outputs)
            get_images_button.click(
                run_class_imb_model_half, inputs=inputs, outputs=img_outputs)
        
        with gr.TabItem("Out of distribution"):
            with gr.Box():
                gr.Markdown(
                    "**Inputs**")
                with gr.Row():
                    gr.Markdown(
                        "Select model and learning rate")
                with gr.Row():
                    model_ood = gr.Dropdown(
                        choices=['LeNet'], value="LeNet", label="Model")
                    learning_rate_ood = gr.Slider(
                        minimum=0.001, maximum=0.1, step=0.001, value=0.01, label="Learning Rate")

                with gr.Row():
                    gr.Markdown("Configure dataset imbalance parameters")

                with gr.Row():
                    datasets_ood = gr.Dropdown(
                        choices=['mnist'], value="mnist", label="Dataset")
                    strategy_ood = gr.Dropdown(
                        choices=['SMI', 'random'], value="SMI", label="Strategy")
                    smi_function_ood = gr.Dropdown(choices=[
                                               'fl2mi', 'gcmi', 'fl1mi'], value="fl2mi", label="SMI function", interactive=True, visible=True)
                    budget_ood = gr.Slider(minimum=10, maximum=1000,
                                       step=10, value=36, label="Budget for labeling")

                with gr.Row():
                    ood_classes = gr.Text(
                        value="0, 1, 2, 3, 4, 5", interactive=True, label="Indistribution classes (Should be in order from 0)")
                    per_ood_lake = gr.Number(
                        value=150, interactive=True, label="Number of lake samples per OOD class")
                with gr.Row():
                    per_class_train_id = gr.Number(
                        value=5, interactive=True, label="Number of train samples per indistribution class")
                    per_class_val_id = gr.Number(
                        value=5, interactive=True, label="Number of validation samples per indistribution class")
                    per_class_lake_id = gr.Number(
                        value=300, interactive=True,  label="Number of lake samples per indistribution class")

                with gr.Row():
                    get_images_button_ood = gr.Button(
                        "Get selected images", variant="primary")
                    submit_button_ood = gr.Button(
                        "Train with selected images", variant="primary")

            with gr.Box():
                gr.Markdown(
                    "**Outputs**")
                with gr.Row():

                    with gr.Column():
                        gain_overall_ood = gr.Textbox(
                            label="Gain in test accuracy")
                        gain_target_ood = gr.Textbox(
                            label="Gain in targeted test accuracy", visible=False)
                        percentage_ood = gr.Textbox(
                            label="Percentage of selected images are in distribution")

                    with gr.Column():
                        imgs_ood = gr.Gallery(label="Selected images for labeling")

                with gr.Row():
                    train_progress_step_1_ood = gr.Timeseries(
                        label="First pass training progress")
                    train_progress_step_2_ood = gr.Timeseries(
                        label="Second pass training progress")

                with gr.Row():
                    initial_metrics_ood = gr.Dataframe(
                        label="Per class accuracies after first pass")
                    final_metrics_ood = gr.Dataframe(
                        label="Per class accuracies after second pass")

            inputs_ood = [model_ood, learning_rate_ood, datasets_ood,
                      strategy_ood, smi_function_ood, budget_ood, ood_classes, per_ood_lake, per_class_train_id, per_class_val_id, per_class_lake_id]

            outputs_ood = [gain_overall_ood, gain_target_ood,
                       initial_metrics_ood, final_metrics_ood, train_progress_step_1_ood, train_progress_step_2_ood, imgs_ood, percentage_ood]

            img_outputs_ood = imgs_ood

            strategy_ood.change(fn=change_strategy_ood,
                            inputs=strategy_ood, outputs=smi_function_ood)
            submit_button_ood.click(run_ood_model,
                                inputs=inputs_ood, outputs=outputs_ood)
            get_images_button_ood.click(
                run_ood_model_half, inputs=inputs_ood, outputs=img_outputs_ood)

        with gr.TabItem("How to use?"):
            gr.Video(value=demo_video_path)
demo.launch(server_port=9000)