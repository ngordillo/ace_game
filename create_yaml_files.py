def create_yaml(run_num): # Open the NetCDF file


    file_template = """
    experiment_dir: ../../Data/ace_output/
    n_forward_steps: 40
    forward_steps_in_memory: 10
    checkpoint_path: ../../Data/ace_data/ckpt/ace_ckpt.tar
    logging:
      log_to_screen: true
      log_to_wandb: false
      log_to_file: true
      project: ace
      # entity: your_wandb_entity
    initial_condition:
      path: ../../Data/ace_output/run""" + str(run_num) + """_restart.nc
      start_indices:
        first: 0
        n_initial_conditions: 1
    forcing_loader:
      dataset:
        data_path: ../../Data/ace_data/validation/ic_0011/
    num_data_workers: 4
    data_writer:
      save_prediction_files: true
      save_monthly_files: false
      names: ["surface_temperature"]
    """.strip()

    for i in range(1,100001):
        file_contents = file_template.format(i)

        with open(f"../configs/default_"+ str(run_num)+ "_test_config.yaml", "w") as f:
            f.write(file_contents)

# create_yaml(1)