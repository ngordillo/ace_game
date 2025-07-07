# ============================
# Create a YAML config file for an ACE experiment run
# ============================
def create_yaml(run_num, fp, ace_fp, episode):  # Open the NetCDF file
    EXP_NAME = "falco_test"

    # ============================
    # Alternate template (commented out)
    # ============================
    # file_template = """
    # experiment_dir: ../../Data/ace_output/
    # n_forward_steps: 200
    # forward_steps_in_memory: 10
    # checkpoint_path: ../../Data/ace_data/ckpt/ace_ckpt.tar
    # logging:
    #   log_to_screen: true
    #   log_to_wandb: false
    #   log_to_file: true
    #   project: ace
    #   # entity: your_wandb_entity
    # initial_condition:
    #   path: ../../Data/ace_output/run""" + str(run_num) + """_restart.nc
    #   start_indices:
    #     first: 0
    #     n_initial_conditions: 1
    # forcing_loader:
    #   dataset:
    #     data_path: ../../Data/ace_data/validation/ic_0011/
    # num_data_workers: 4
    # data_writer:
    #   save_prediction_files: true
    #   save_monthly_files: false
    #   names: ["surface_temperature"]
    # """.strip()

    # ============================
    # YAML config template to be written (based on HPC setup)
    # ============================
    file_template = """
    experiment_dir: /GPU-scratch/nicojg/ace/ace_output/
    n_forward_steps: 200
    forward_steps_in_memory: 10
    checkpoint_path: /GPU-scratch/nicojg/ace/ace_data/ace_ckpt.tar
    logging:
      log_to_screen: true
      log_to_wandb: false
      log_to_file: true
      project: ace
      # entity: your_wandb_entity
    initial_condition:
      path: /GPU-scratch/nicojg/ace/ace_output/run""" + str(run_num) + """_restart.nc
      start_indices:
        first: 0
        n_initial_conditions: 1
    forcing_loader:
      dataset:
        data_path: /GPU-scratch/nicojg/ace/ace_data/fv3/validation/ic_0011
    num_data_workers: 4
    data_writer:
      save_prediction_files: true
      save_monthly_files: false
      names: ["surface_temperature"]
    """.strip()

    # ============================
    # Redundant commented-out example YAML below
    # ============================
    #     n_forward_steps: 200
    # forward_steps_in_memory: 10
    # checkpoint_path: /GPU-scratch/nicojg/ace/ace_data/ace_ckpt.tar
    # logging:
    #   log_to_screen: true
    #   log_to_wandb: false
    #   log_to_file: true
    #   project: ace
    #   entity: your_wandb_entity
    # initial_condition:
    #   path: /GPU-scratch/nicojg/ace/ace_data/initialization/initialization_data.nc
    # forcing_loader:
    #   dataset:
    #     data_path: /GPU-scratch/nicojg/ace/ace_data/fv3/validation/ic_0011
    #   num_data_workers: 4
    # data_writer:
    #   save_prediction_files: true
    #   save_monthly_files: false

    # ============================
    # Debug print statements
    # ============================
    print("PEISODCFSFSDFDS")  # Likely a typo or test marker
    print(episode)
    print("sdafdsfsfdsfddas")

    # ============================
    # Write YAML config file for the given run number and episode
    # ============================
    for i in range(1, 100001):  # Unused loop index `i` – template does not use `.format(i)`
        file_contents = file_template.format(i)

        with open(ace_fp + f"configs/" + EXP_NAME + "/episode" + str(episode) + "/" + str(run_num) + "_test_config.yaml", "w") as f:
            f.write(file_contents)

# create_yaml(1)  # Call left commented out
