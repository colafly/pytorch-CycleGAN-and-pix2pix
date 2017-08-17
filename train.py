import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from zipfile import ZipFile
from tinyenv.flags import flags
from os import listdir

FLAGS = flags()

z = ZipFile(FLAGS.dataroot +  FLAGS.dataset_name_zip)
z.extractall(FLAGS.dataroot)
print(listdir(FLAGS.dataroot))

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)

total_steps = 0

for epoch in range(1, opt.iterations + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            if opt.display_id > 0:
                print(epoch, float(epoch_iter)/dataset_size)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.iterations + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.iterations:
        model.update_learning_rate()
