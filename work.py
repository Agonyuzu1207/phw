import subprocess

commandICEWS14_data = [
    "python", "preprocess_data.py", "--data_dir", "data/ICEWS14","-jump=2", "-maxN=40","-padding=50",
]
commandICEWS18_data = [
    "python", "preprocess_data.py", "--data_dir", "data/ICEWS18","-jump=2", "-maxN=40","-padding=50",
    # "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS05-15"
]
commandICEWS05_data = [
    "python", "preprocess_data.py", "--data_dir", "data/ICEWS05-15","-jump=2", "-maxN=40","-padding=50",
    # "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS05-15"
]
commandWIKI_data = [
    "python", "preprocess_data.py", "--data_dir", "data/WIKI","-jump=2", "-maxN=40","-padding=60",
    # "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS05-15"
]
commandYAGO_data = [
    "python", "preprocess_data.py", "--data_dir", "data/YAGO","-jump=2", "-maxN=40","-padding=30",
    # "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS05-15"
]
commandGDELT_data = [
    "python", "preprocess_data.py", "--data_dir", "data/GDELT","-jump=2", "-maxN=40","-padding=30",
    # "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS05-15"
]


commandICEWS14_work = [
    "python", "main.py", "--data_path", "data/ICEWS14", "--batch_size=128","--test_batch_size=64",
    "--cuda","--do_test", "--jump=2", "--padding=50", "--savestep=5", "--desc=ICEWS14"
    ,"--do_train",
]
commandICEWS18_work = [
    "python", "main.py", "--data_path", "data/ICEWS18","--batch_size=4","--test_batch_size=4",
    "--cuda","--do_test", "--jump=2", "--padding=50", "--savestep=5", "--desc=ICEWS18"
    ,"--do_train",
]
commandICEWS05_work = [
    "python", "main.py", "--data_path", "data/ICEWS05-15","--batch_size=4","--test_batch_size=1",
    "--cuda","--do_test", "--jump=2", "--padding=50", "--savestep=5", "--desc=ICEWS05-15"
    ,"--do_train",
]
commandWIKI_work = [
    "python", "main.py", "--data_path", "data/WIKI","--batch_size=4","--test_batch_size=4",
    "--cuda","--do_test", "--jump=2", "--padding=60", "--savestep=5", "--desc=WIKI"
    ,"--do_train",
]
commandYAGO_work = [
    "python", "main.py", "--data_path", "data/YAGO","--batch_size=8","--test_batch_size=4",
    "--cuda","--do_test", "--jump=2", "--padding=30", "--savestep=5", "--desc=YAGO"
    ,"--do_train",
]
commandGDELT_work = [
    "python", "main.py", "--data_path", "data/GDELT","--batch_size=8","--test_batch_size=4",
    "--cuda","--do_test", "--jump=2", "--padding=30", "--savestep=5", "--desc=GDELT"
    ,"--do_train",
]

commandICEWS14_test = [
    "python", "main.py", "--data_path", "data/ICEWS14", "--load_model_path=logs/ICEWS14/3_transformer/j2_20241117_21_13_52/checkpoint.pth",
    "--test_batch_size=1","--cuda","--do_test", "--jump=2", "--padding=50", "--savestep=5", "--desc=ICEWS14"
]
commandICEWS18_test = [
    "python", "main.py", "--data_path", "data/ICEWS18","--batch_size=5","--load_model_path=logs/ICEWS18/common/j2_20241030_12_21_51/checkpoint_2.pth",
    "--cuda","--do_test", "--jump=2", "--padding=50", "--savestep=5", "--desc=ICEWS18",
]
commandICEWS05_test = [
    "python", "main.py", "--data_path", "data/ICEWS05-15","--batch_size=5","--load_model_path=logs/ICEWS05-15/3_MLP/j2_20241104_20_10_28/checkpoint_2.pth",
    "--cuda","--do_test", "--jump=2", "--padding=50", "--savestep=5","--desc=ICEWS05-15",
]
commandWIKI_test = [
    "python", "main.py", "--data_path", "data/WIKI","--batch_size=5","--load_model_path=logs/WIKI/common/j2_20241031_15_05_34/checkpoint_2.pth",
    "--cuda","--do_test", "--jump=2", "--padding=60", "--savestep=5", "--desc=WIKI",
]
commandYAGO_test = [
    "python", "main.py", "--data_path", "data/YAGO","--batch_size=10","--load_model_path=logs/YAGO/common/j2_20241101_09_10_36/checkpoint_2.pth",
    "--cuda","--do_test", "--jump=2", "--padding=30", "--savestep=5", "--desc=YAGO",
]
commandGDELT_test = [
    "python", "main.py", "--data_path", "data/GDELT","--batch_size=10","--load_model_path=logs/YAGO/common/j2_20241101_09_10_36/checkpoint_2.pth",
    "--cuda","--do_test", "--jump=2", "--padding=50", "--savestep=5", "--desc=GDELT",
]




commandICEWS14_work1 = [
    "python", "main.py", "--data_path", "data/ICEWS14", "--batch_size=16","--test_batch_size=4","--path_length=2",
    "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS14"
    ,"--do_train",
]
commandICEWS14_work2 = [
    "python", "main.py", "--data_path", "data/ICEWS14", "--batch_size=2","--test_batch_size=4","--path_length=4",
    "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS14"
    ,"--do_train",
]
commandICEWS14_work3 = [
    "python", "main.py", "--data_path", "data/ICEWS14", "--batch_size=1","--test_batch_size=4","--path_length=5",
    "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS14"
    ,"--do_train",
]

commandICEWS18_work1 = [
    "python", "main.py", "--data_path", "data/ICEWS18", "--batch_size=16","--test_batch_size=4", "--path_length=2",
    "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS18"
    ,"--do_train",
]
commandICEWS18_work2 = [
    "python", "main.py", "--data_path", "data/ICEWS18", "--batch_size=8","--test_batch_size=4","--path_length=4",
    "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS18"
    ,"--do_train",
]
commandICEWS18_work3 = [
    "python", "main.py", "--data_path", "data/ICEWS18", "--batch_size=8","--test_batch_size=4","--path_length=5",
    "--cuda","--do_test", "--jump=2", "--padding=140", "--savestep=5", "--desc=ICEWS18"
    ,"--do_train",
]

# subprocess.run(commandICEWS14_data)
# subprocess.run(commandICEWS18_data)
# subprocess.run(commandICEWS05_data)
# subprocess.run(commandWIKI_data)
# subprocess.run(commandYAGO_data)
# subprocess.run(commandGDELT_data)


# subprocess.run(commandICEWS14_test)
# subprocess.run(commandICEWS18_test)
# subprocess.run(commandICEWS05_test)
# subprocess.run(commandWIKI_test)
# subprocess.run(commandYAGO_test)


subprocess.run(commandICEWS14_work)
# subprocess.run(commandICEWS18_work)
# subprocess.run(commandWIKI_work)
# subprocess.run(commandYAGO_work)
# subprocess.run(commandICEWS05_work)
# subprocess.run(commandGDELT_work)

 