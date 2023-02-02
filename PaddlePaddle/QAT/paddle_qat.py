import paddle
import paddleslim as slim
import numpy as np
paddle.enable_static()

USE_GPU = False
model = slim.models.ResNet50()
train_program = paddle.static.Program()
startup = paddle.static.Program()
with paddle.static.program_guard(train_program, startup):
    image = paddle.static.data(
        name='image', shape=[None, 1, 28, 28], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    gt = paddle.reshape(label, [-1, 1])
    out = model.net(input=image, class_dim=10)
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=gt)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=gt, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=gt, k=5)
    opt = paddle.optimizer.Momentum(0.01, 0.9)
    opt.minimize(avg_cost)

place = paddle.CUDAPlace(0) if USE_GPU else paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(startup)
val_program = train_program.clone(for_test=True)

# train
import paddle.vision.transforms as T
transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
train_dataset = paddle.vision.datasets.MNIST(
    mode="train", backend="cv2", transform=transform)
test_dataset = paddle.vision.datasets.MNIST(
    mode="test", backend="cv2", transform=transform)
train_loader = paddle.io.DataLoader(
    train_dataset,
    places=place,
    feed_list=[image, label],
    drop_last=True,
    batch_size=64,
    return_list=False,
    shuffle=True)
test_loader = paddle.io.DataLoader(
    test_dataset,
    places=place,
    feed_list=[image, label],
    drop_last=True,
    batch_size=64,
    return_list=False,
    shuffle=False)


# tests
outputs = [acc_top1.name, acc_top5.name, avg_cost.name]
def train(prog):
    iter = 0
    for data in train_loader():
        acc1, acc5, loss = exe.run(prog, feed=data, fetch_list=outputs)
        if iter % 100 == 0:
            print('train iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        iter += 1

def test(prog):
    iter = 0
    res = [[], []]
    for data in test_loader():
        acc1, acc5, loss = exe.run(prog, feed=data, fetch_list=outputs)
        if iter % 100 == 0:
            print('test iter={}, top1={}, top5={}, loss={}'.format(iter, acc1.mean(), acc5.mean(), loss.mean()))
        res[0].append(acc1.mean())
        res[1].append(acc5.mean())
        iter += 1
    print('final test result top1={}, top5={}'.format(np.array(res[0]).mean(), np.array(res[1]).mean()))


train(train_program)
test(val_program)

# # quantization

quant_program = slim.quant.quant_aware(train_program, exe.place, for_test=False)
val_quant_program = slim.quant.quant_aware(val_program, exe.place, for_test=True)

train(quant_program)
test(val_quant_program)

quant_infer_program = slim.quant.convert(val_quant_program, exe.place)
target_vars = [quant_infer_program.global_block().var(outputs[-1])]
paddle.static.save_inference_model(
        path_prefix='./quant_infer_model',
        feed_vars=[image],
        fetch_vars=target_vars,
        executor=exe,
        program=quant_infer_program)

# save to onnx
# save_path = './MobileNet' # 需要保存的路径
# x_spec = paddle.static.InputSpec([None, 1, 28, 28], 'float32', 'image')
# paddle.onnx.export(model, save_path, input_spec=[x_spec], opset_version=11)

# paddle2onnx --model_dir model/ --model_filename quant_infer_model.pdmodel 
# --params_filename quant_infer_model.pdiparams --save_file model/quant_infer_model.onnx
#  --enable_onnx_checker True --opset_version 16 --deploy_backend others

