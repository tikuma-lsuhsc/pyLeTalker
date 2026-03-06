import letalker as lt


def test_add_aspiration_noise():
    # self, noise_model: AspirationNoise | bool | dict | None

    vf = lt.VocalFoldsUg(lt.SineGenerator(129))
    vf.add_aspiration_noise(None)
    assert vf.noise_model is None
    vf.add_aspiration_noise(False)
    assert vf.noise_model is None
    vf.add_aspiration_noise(True)
    assert isinstance(vf.noise_model, lt.LeTalkerAspirationNoise)
    vf.add_aspiration_noise(lt.LeTalkerAspirationNoise())
    assert isinstance(vf.noise_model, lt.LeTalkerAspirationNoise)
    vf.add_aspiration_noise({"noise_source": lt.WhiteNoiseGenerator()})
    assert isinstance(vf.noise_model, lt.LeTalkerAspirationNoise)


def test_create_noise_runner():

    vf = lt.VocalFoldsUg(lt.SineGenerator(129))
    vf.add_aspiration_noise(None)
    vf.create_noise_runner(100)


def test_areas_of_connected_tracts(): ...


def test_configure_resistances(): ...


def test_create_runner(): ...


def test_create_results(): ...
