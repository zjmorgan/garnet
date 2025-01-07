from garnet.config.instruments import beamlines
from garnet.reduction.data import DataModel

def test_white():

    corelli = DataModel(beamlines['CORELLI'])

    assert corelli.laue == True

    mandi = DataModel(beamlines['MANDI'])

    assert mandi.laue == True

    snap = DataModel(beamlines['SNAP'])

    assert snap.laue == True

    topaz = DataModel(beamlines['TOPAZ'])

    assert topaz.laue == True

def test_monochromatic():

    demand = DataModel(beamlines['DEMAND'])

    assert demand.laue == False

    wand2 = DataModel(beamlines['WAND²'])

    assert wand2.laue == False