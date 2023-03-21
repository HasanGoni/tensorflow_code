import matplotlib.pyplot as plt
from pathlib import Path


def show_batches(ds, size=9):
    """
    display number of images
    from a dataset
    """
    ds = ds.unbatch().take(size)
    n_cols = 3
    n_rows = size // n_cols + 1
    plt.figure(figsize=(18, 10))
    i = 0
    for im in ds:
        i += 1
        im = im.numpy()
        # print(im.shape)
        plt.subplot(n_rows, n_cols, i)
        plt.axis('off')
        plt.imshow(im)
    plt.show()


def generate_and_save_images(model,
                             epoch,
                             step,
                             test_input):
    """
    """
    path = Path.cwd()
    path = Path(path/'images_anime')
    path.mkdir(exist_ok=True,
               parents=True)
    predictions = model.predict(test_input)
    fig = plt.figure(figsize=(5, 5))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        img = predictions[i, :, :, :] * 255
        img = img.astype('int32')
        plt.imshow(img)

        plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
    fig.suptitle("epoch: {}, step: {}".format(epoch, step))
    plt.savefig(path/f'image_at_epoch_{epoch:04d}_step{step:04d}.png')

    # plt.show()
