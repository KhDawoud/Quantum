"""Fast correctness tests for src/model_vqc.py (fake data scaled to [0, pi])."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.model_vqc import PARAM_GRID, VQCClassifier, build_qnode, make_model, zz_feature_map


def fake_data(n=60, n_features=4, seed=0):
    """Two overlapping blobs, min-max scaled into [0, pi] like PimaPreprocessor does."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2), np.ones(n - n // 2)].astype(int)
    X = rng.normal(0, 1, (n, n_features)) + y[:, None] * 1.5
    X = (X - X.min(0)) / (X.max(0) - X.min(0)) * np.pi
    return X, y


def test_param_grid_size_and_keys():
    n_combos = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    assert n_combos <= 4
    assert set(PARAM_GRID) <= set(make_model().get_params())


def test_isingzz_equals_cnot_rz_cnot_feature_map():
    """The speed-up (IsingZZ instead of CNOT-RZ-CNOT) must not change the quantum state."""
    n = 4
    dev = qml.device("default.qubit", wires=n)
    x = np.random.default_rng(1).uniform(0, np.pi, (5, n))

    @qml.qnode(dev)
    def fast(x):
        zz_feature_map(x, n)
        return qml.state()

    @qml.qnode(dev)
    def spec(x):  # literal PROJECT_CONTEXT.md description
        for _ in range(2):
            for i in range(n):
                qml.Hadamard(wires=i)
            for i in range(n):
                qml.RZ(x[:, i], wires=i)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(x[:, i] * x[:, i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        return qml.state()

    np.testing.assert_allclose(fast(x), spec(x), atol=1e-10)


@pytest.mark.parametrize("encoding", ["zz", "angle"])
def test_batched_equals_one_patient_at_a_time(encoding):
    n = 3
    circuit = build_qnode(n, encoding)
    rng = np.random.default_rng(2)
    X = rng.uniform(0, np.pi, (6, n))
    w = rng.normal(size=qml.StronglyEntanglingLayers.shape(2, n))
    batched = np.asarray(circuit(w, X))
    looped = np.array([float(circuit(w, X[i : i + 1])[0]) for i in range(len(X))])
    np.testing.assert_allclose(batched, looped, atol=1e-10)


@pytest.mark.parametrize("encoding", ["zz", "angle"])
def test_fit_predict_shapes_and_probabilities(encoding):
    X, y = fake_data()
    model = make_model(encoding=encoding, epochs=2).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(model.predict(X))) <= {0, 1}
    assert len(model.loss_history_) == 2
    assert model.weights_.shape == (2, X.shape[1], 3)


def test_loss_history_length_matches_epochs():
    X, y = fake_data()
    assert len(make_model(epochs=5).fit(X, y).loss_history_) == 5


def test_same_seed_same_weights():
    X, y = fake_data()
    a = make_model(epochs=3).fit(X, y)
    b = make_model(epochs=3).fit(X, y)
    np.testing.assert_array_equal(a.weights_, b.weights_)
    assert a.loss_history_ == b.loss_history_


def test_rejects_inputs_outside_zero_pi():
    X, y = fake_data()
    with pytest.raises(ValueError, match="0, pi"):
        make_model(epochs=1).fit(X * 2.0, y)


def test_class_weights_favour_the_rare_class():
    X, y = fake_data(n=60)
    y = (np.arange(60) < 20).astype(int)  # 20 positives, 40 negatives
    model = make_model(epochs=1).fit(X, y)
    assert model.class_weights_[1] > model.class_weights_[0]
    np.testing.assert_allclose(model.class_weights_, [60 / 80, 60 / 40])


def test_balanced_class_weights_computed_from_the_labels_given_to_fit():
    """Weights = n / (2 * class count), recomputed from whatever y fit() receives."""
    X, _ = fake_data(n=10, n_features=2)

    # 7 healthy (0) and 3 diabetic (1): 10/(2*7) = 0.714..., 10/(2*3) = 1.666...
    y_7_3 = np.array([0] * 7 + [1] * 3)
    w = make_model(epochs=1).fit(X, y_7_3).class_weights_
    np.testing.assert_allclose(w, [10 / 14, 10 / 6])
    assert w[0] == pytest.approx(0.714, abs=1e-3)
    assert w[1] == pytest.approx(1.667, abs=1e-3)

    # A different label mix (same X): weights must change, so they are not hard-coded.
    y_5_5 = np.array([0] * 5 + [1] * 5)
    w55 = make_model(epochs=1).fit(X, y_5_5).class_weights_
    np.testing.assert_allclose(w55, [1.0, 1.0])
    y_9_1 = np.array([0] * 9 + [1])
    w91 = make_model(epochs=1).fit(X, y_9_1).class_weights_
    np.testing.assert_allclose(w91, [10 / 18, 10 / 2])

    # Same ratio as our real data (500 healthy / 268 diabetic): about 1.87.
    assert (w[1] / w[0]) == pytest.approx(7 / 3)  # ratio = count ratio
    assert (768 / (2 * 268)) / (768 / (2 * 500)) == pytest.approx(500 / 268)


def test_sklearn_clone_and_gridsearch():
    X, y = fake_data(n=40, n_features=2)
    base = make_model(epochs=1)
    assert clone(base).get_params() == base.get_params()
    grid = {"lr": [0.02, 0.05]}
    search = GridSearchCV(
        base, grid, cv=StratifiedKFold(2, shuffle=True, random_state=42), scoring="balanced_accuracy"
    ).fit(X, y)
    assert search.best_params_["lr"] in grid["lr"]


def test_loss_goes_down_on_easy_data():
    X, y = fake_data(n=60, n_features=2)
    model = make_model(encoding="angle", epochs=15, lr=0.05).fit(X, y)
    assert model.loss_history_[-1] < model.loss_history_[0]


# --------------------------------------------------------------------------- #
# Experiment 6: final_rotation option
# --------------------------------------------------------------------------- #
# Recorded with the code BEFORE final_rotation existed (commit 4a7f922), ZZ encoding,
# fake_data(n=40, n_features=3, seed=5), epochs=3, lr=0.05. Bit-identical on the
# machine where it was recorded; atol=1e-12 only allows for other CPUs/BLAS builds.
REF_ZZ_WEIGHTS = np.array([
    -0.23049375098156538, -0.39488258044733654, 0.01051315423265046,
    -0.02579565856442932, -0.2382568471309957, -0.2560875740167479,
    -0.2613484321463911, -0.30746715550820974, -0.17559604512316052,
    -0.10763936546215311, -0.19110183103901818, 0.07777919363711365,
    -0.02069388571872444, -0.0991133679955066, 0.04675093421377267,
    -0.08592924624836919, 0.03687507835437991, -0.09588826004271137,
]).reshape(2, 3, 3)
REF_ZZ_P1_FIRST5 = np.array([0.27509307818170514, 0.40584227523907235, 0.4564595419763017,
                             0.2571023579102386, 0.5797031373806039])
REF_ZZ_LOSS = [0.7443897265768971, 0.6829482830925295, 0.6375032784015067]


def test_final_rotation_off_reproduces_original_model():
    X, y = fake_data(n=40, n_features=3, seed=5)
    default = make_model(encoding="zz", epochs=3, lr=0.05).fit(X, y)
    explicit = make_model(encoding="zz", epochs=3, lr=0.05, final_rotation=False).fit(X, y)
    # Default and explicit False: exactly the same numbers.
    np.testing.assert_array_equal(default.weights_, explicit.weights_)
    np.testing.assert_array_equal(default.predict_proba(X), explicit.predict_proba(X))
    assert default.loss_history_ == explicit.loss_history_
    assert not hasattr(default, "final_weights_")
    # Same numbers as the code before the option existed.
    np.testing.assert_allclose(default.weights_, REF_ZZ_WEIGHTS, rtol=0, atol=1e-12)
    np.testing.assert_allclose(default.predict_proba(X)[:5, 1], REF_ZZ_P1_FIRST5,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(default.loss_history_, REF_ZZ_LOSS, rtol=0, atol=1e-12)


def test_final_rotation_draws_extra_angles_after_main_weights():
    """With 0 epochs nothing is trained: the main start weights must match the default."""
    X, y = fake_data(n=40, n_features=3, seed=5)
    off = make_model(epochs=0).fit(X, y)
    on = make_model(epochs=0, final_rotation=True).fit(X, y)
    np.testing.assert_array_equal(off.weights_, on.weights_)
    rng = np.random.default_rng(42)
    rng.normal(0.0, 0.1, size=off.weights_.shape)
    np.testing.assert_array_equal(on.final_weights_, rng.normal(0.0, 0.1, size=3))


def test_final_rotation_trains_changes_output_and_pickles():
    import pickle

    X, y = fake_data(n=40, n_features=3, seed=5)
    start = make_model(epochs=0, final_rotation=True).fit(X, y).final_weights_
    on = make_model(epochs=3, lr=0.05, final_rotation=True).fit(X, y)
    off = make_model(epochs=3, lr=0.05).fit(X, y)
    assert on.final_weights_.shape == (3,)
    assert not np.allclose(on.final_weights_, start)  # the 3 angles were trained
    assert len(on.loss_history_) == 3
    assert not np.allclose(on.predict_proba(X), off.predict_proba(X))
    proba = on.predict_proba(X)
    assert np.all((proba >= 0) & (proba <= 1))
    restored = pickle.loads(pickle.dumps(on))
    np.testing.assert_array_equal(restored.predict_proba(X), proba)
    assert clone(on).get_params()["final_rotation"] is True


@pytest.mark.parametrize("encoding", ["zz", "angle"])
def test_autograd_gradient_matches_finite_differences(encoding):
    """Training only works if the backprop gradient is right: compare with (f(w+h)-f(w-h))/2h."""
    from pennylane import numpy as pnp

    n = 3
    circuit = build_qnode(n, encoding)
    rng = np.random.default_rng(3)
    X = rng.uniform(0, np.pi, (8, n))
    w0 = rng.normal(size=qml.StronglyEntanglingLayers.shape(2, n))

    def f(w):
        return pnp.mean(circuit(w, X) ** 2)

    w = pnp.array(w0, requires_grad=True)
    auto = np.asarray(qml.grad(f)(w))
    numeric = np.zeros_like(w0)
    h = 1e-5
    for idx in np.ndindex(*w0.shape):
        up, dn = w0.copy(), w0.copy()
        up[idx] += h
        dn[idx] -= h
        numeric[idx] = (float(f(pnp.array(up, requires_grad=False)))
                        - float(f(pnp.array(dn, requires_grad=False)))) / (2 * h)
    np.testing.assert_allclose(auto, numeric, atol=1e-6)
