class StructureFactorCalculator:

    def __init__(self, view, model):

        self.view = view
        self.model = model

        self.view.crystal_system_combo.activated.connect(self.generate_groups)
        self.view.space_group_combo.activated.connect(self.generate_settings)

        self.view.load_CIF_button.clicked.connect(self.load_CIF)

        self.generate_groups()
        self.generate_settings()

    def generate_groups(self):

        system = self.get_crystal_system()
        nos = self.model.generate_space_groups_from_crystal_system(system)
        self.update_space_groups(nos)

    def generate_settings(self):

        no = self.get_space_group()
        settings = self.model.generate_settings_from_space_group(no)
        self.update_space_groups(settings)


    def load_CIF(self):

        pass