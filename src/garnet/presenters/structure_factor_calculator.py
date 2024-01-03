class StructureFactorCalculator:

    def __init__(self, view, model):

        self.view = view
        self.model = model

        self.view.crystal_system_combo.activated.connect(self.generate_groups)
        self.view.space_group_combo.activated.connect(self.generate_settings)
        self.view.calculate_button.clicked.connect(self.calculate_F2)

        self.view.load_CIF_button.clicked.connect(self.load_CIF)

        self.view.manual_button.clicked.connect(self.view_manual)

        self.view.a_star_button.clicked.connect(self.view_bc_star)
        self.view.b_star_button.clicked.connect(self.view_ca_star)
        self.view.c_star_button.clicked.connect(self.view_ab_star)

        self.view.a_button.clicked.connect(self.view_bc)
        self.view.b_button.clicked.connect(self.view_ca)
        self.view.c_button.clicked.connect(self.view_ab)

        self.view.view_combo.currentIndexChanged.connect(self.update_labels)
        self.view.proj_box.clicked.connect(self.change_proj)
        self.view.reset_button.clicked.connect(self.reset_view)

        self.generate_groups()
        self.generate_settings()

    def update_labels(self):

        self.view.update_axis_labels()

    def change_proj(self):

        self.view.change_proj()

    def reset_view(self):

        self.view.reset_view()

    def view_ab_star(self):

        vecs = self.model.ab_star_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_bc_star(self):

        vecs = self.model.bc_star_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_ca_star(self):
        vecs = self.model.ca_star_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_ab(self):

        vecs = self.model.ab_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_bc(self):

        vecs = self.model.bc_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_ca(self):

        vecs = self.model.ca_axes()
        if vecs is not None:
            self.view.view_vector(vecs)

    def view_manual(self):

        indices = self.view.get_manual_indices()

        if indices is not None:
            vec = self.model.get_vector(*indices)
            if vec is not None:
                self.view.view_vector(vec)

    def generate_groups(self):

        system = self.view.get_crystal_system()
        nos = self.model.generate_space_groups_from_crystal_system(system)
        self.view.update_space_groups(nos)

        self.generate_settings()

    def generate_settings(self):

        no = self.view.get_space_group()
        settings = self.model.generate_settings_from_space_group(no)
        self.view.update_settings(settings)

    def load_CIF(self):

        filename = self.view.load_CIF_file_dialog()

        if filename:

            self.model.load_CIF(filename)

            crystal_system = self.model.get_crystal_system()
            space_group = self.model.get_space_group()
            setting = self.model.get_setting()
            params = self.model.get_lattice_constants()
            scatters = self.model.get_scatterers()

            self.view.set_crystal_system(crystal_system)
            self.generate_groups()
            self.view.set_space_group(space_group) 
            self.generate_settings()
            self.view.set_setting(setting)
            self.view.set_lattice_constants(params)
            self.view.set_scatterers(scatters)

            params = self.model.constrain_parameters()
            self.view.constrain_parameters(params)

            atom_dict = self.model.generate_atom_positions()
            self.view.add_atoms(atom_dict)

            self.view.draw_cell(self.model.get_unit_cell_transform())
            self.view.set_transform(self.model.get_transform())

    def calculate_F2(self):

        dmin = self.view.get_minimum_d_spacing()

        params = self.view.get_lattice_constants()

        hkls, ds, F2s = self.model.generate_F2(dmin)