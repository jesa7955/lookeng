from logging import getLogger

import gokart

logger = getLogger(__name__)


class Sample(gokart.TaskOnKart):
    task_namespace = 'm3lookeng'

    def output(self):
        return self.make_target('data/sample.pkl')

    def run(self):
        self.dump('sample output')
