

def load_training_data(dataFile, validation_split=0, axes=None, n_images=None, verbose=False):

    f = np.load(dataFile)
    X, Y = f['X'], f['Y']
    if axes is None:
        axes = f['axes']
    axes = axes_check_and_normalize(axes)

    assert X.shape == Y.shape
    assert len(axes) == X.ndim
    assert 'C' in axes
    if n_images is None:
        n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
    assert 0 <= validation_split < 1

    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']

    if validation_split > 0:
        n_val   = int(round(n_images * validation_split))
        n_train = n_images - n_val
        assert 0 < n_val and 0 < n_train
        X_t, Y_t = X[-n_val:],  Y[-n_val:]
        X,   Y   = X[:n_train], Y[:n_train]
        assert X.shape[0] == n_train and X_t.shape[0] == n_val
        X_t = move_channel_for_backend(X_t,channel=channel)
        Y_t = move_channel_for_backend(Y_t,channel=channel)

    X = move_channel_for_backend(X,channel=channel)
    Y = move_channel_for_backend(Y,channel=channel)

    axes = axes.replace('C','') # remove channel
    if backend_channels_last():
        axes = axes+'C'
    else:
        axes = axes[:1]+'C'+axes[1:]

    data_val = (X_t,Y_t) if validation_split > 0 else None

    return (X,Y), data_val, axes



def save_training_data(file, X, Y, axes):
    """Save training data in ``.npz`` format.

    Parameters
    ----------
    file : str
        File name
    X : :class:`numpy.ndarray`
        Array of patches extracted from source images.
    Y : :class:`numpy.ndarray`
        Array of corresponding target patches.
    axes : str
        Axes of the extracted patches.

    """
    isinstance(file,(Path,string_types)) or _raise(ValueError())
    file = Path(file).with_suffix('.npz')
    file.parent.mkdir(parents=True,exist_ok=True)

    axes = axes_check_and_normalize(axes)
    len(axes) == X.ndim or _raise(ValueError())
    np.savez(str(file), X=X, Y=Y, axes=axes)
