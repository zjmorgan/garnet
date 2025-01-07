from mantid.geometry import PointGroupFactory, SpaceGroupFactory

point_groups = PointGroupFactory.getAllPointGroupSymbols()
space_groups = SpaceGroupFactory.getAllSpaceGroupSymbols()

point_crystal = {}
point_lattice = {}
point_laue = {}

for point_group in point_groups:
    pg = PointGroupFactory.createPointGroup(point_group)
    pg_name = pg.getHMSymbol().replace(' ','')
    point_crystal[pg_name] = pg.getCrystalSystem().name
    point_lattice[pg_name] = pg.getLatticeSystem().name
    point_laue[pg_name] = pg.getLauePointGroupSymbol()

space_point = {}
space_number = {}

for space_group in space_groups:
    sg = SpaceGroupFactory.createSpaceGroup(space_group)
    sg_name = sg.getHMSymbol().replace(' ','')
    space_point[sg_name] = sg.getPointGroup().getHMSymbol().replace(' ','')
    space_number[sg_name] = sg.getNumber()
