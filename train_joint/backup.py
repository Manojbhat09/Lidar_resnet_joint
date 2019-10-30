	# Create log file
try:
	if os.path.isdir(FLAGS.log):
		shutil.rmtree(FLAGS.log)
	os.makedirs(FLAGS.log)
except Exception as e:
	print(e)
	print("Error creating log directory. ")
	quit()

	# Checking pretrained model 
if FLAGS.pretrained is not None:
	if os.path.isdir(FLAGS.pretrained):
		print("Model folder exists, using model from %s" % (FLAGS.pretrained))
	else:
		print("Model folder doesnt exists, starting with random weights ")
else:
	print("No pretrained model, using random weights ")

	# Making backup of important files
try:
	print("Copying important files ...")
	copyfile(FLAGS.arch_cfg, FLAGS.log+"/arch_cfg.yaml")
	copyfile(FLAGS.data_cfg, FLAGS.log+"/data_cfg.yaml")
except Exception as e:
	print(e)
	print("Error copying files ")
	quit() 