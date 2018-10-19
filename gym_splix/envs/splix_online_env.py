from itertools import product

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.firefox.options import Options
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imread
from gym import spaces, Env

FIREFOX_PATH = 'C:\\Program Files\\Mozilla Firefox\\firefox.exe'


class SplixOnlineEnv(Env):
    """ Env connected to splix.io, thanks to Selenium (on firefox driver)

    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    __PLAYER_NAME = "Blublu"
    __MAP_SIZE = 600
    __VIEW_SIZE = 51
    __FIRST_SCORE = 25
    __DIRECTIONS = {'RIGHT': 0, 'DOWN': 1, 'LEFT': 2, 'UP': 3, 'PAUSE': 4}
    __FEATURES_CHANNELS = {'myBlocks': 0, 'freeSpace': 1, 'myTrail': 2, 'otherTrail': 3, "otherPositions": 4,
                           'outside': 5}
    __ACTIONS = {'GoStraight': 0, 'GoLeft': 1, 'GoRight': 2, 'Pause': 3}

    observation_space = spaces.Box(low=0, high=1.0, shape=(__VIEW_SIZE, __VIEW_SIZE, len(__FEATURES_CHANNELS)),
                                   dtype=np.bool)
    action_space = spaces.Discrete(len(__ACTIONS))

    def __init__(self, firefox_path=FIREFOX_PATH):
        """

        :param firefox_path: your own firefox path
        """
        # driver
        self._driver = None
        self.firefox_path=firefox_path

        # your info in the game
        self._my_block_id = None
        self._my_blocks = None
        self._my_position = None
        self._free_blocks = None
        self._my_trails = None
        self._other_trails = None
        self._other_positions = None

        # info about your score
        self._score = self.__FIRST_SCORE
        self._is_dead = None
        self._rank = None
        self._captured_blocks = None
        self._kills = None

        self._observation = None

        self.viewer=None

    ##########################
    #    Gym Env Methods
    ##########################

    def step(self, action):
        """

        :param action: from __ACTIONS space
        :return: obs, reward, done, info
        """

        old_score = self._score

        dom = self._driver.find_element_by_tag_name("html")

        if action == self.__ACTIONS['GoStraight']:
            if self._my_direction == self.__DIRECTIONS['PAUSE']:
                dom.send_keys(Keys.UP)

        if action == self.__ACTIONS['GoLeft']:
            if self._my_direction == self.__DIRECTIONS['RIGHT']:
                dom.send_keys(Keys.UP)
            elif self._my_direction == self.__DIRECTIONS['DOWN']:
                dom.send_keys(Keys.RIGHT)
            elif self._my_direction == self.__DIRECTIONS['LEFT']:
                dom.send_keys(Keys.DOWN)
            elif self._my_direction == self.__DIRECTIONS['UP']:
                dom.send_keys(Keys.LEFT)
            else:  # Pause
                dom.send_keys(Keys.LEFT)

        elif action == self.__ACTIONS['GoRight']:
            if self._my_direction == self.__DIRECTIONS['RIGHT']:
                dom.send_keys(Keys.DOWN)
            elif self._my_direction == self.__DIRECTIONS['DOWN']:
                dom.send_keys(Keys.LEFT)
            elif self._my_direction == self.__DIRECTIONS['LEFT']:
                dom.send_keys(Keys.UP)
            elif self._my_direction == self.__DIRECTIONS['UP']:
                dom.send_keys(Keys.RIGHT)
            else:  # Pause
                dom.send_keys(Keys.RIGHT)

        elif action == self.__ACTIONS['Pause']:
            dom.send_keys("P")

        obs = self.observation
        if self.done:
            reward = - self._score / 2
        else:
            reward = self._score - old_score
        done = self.done

        info = {
            'score': self._score,
            'kills': self._kills,
            'rank': self._rank,
            'captured_blocks': self._captured_blocks
        }

        return obs, reward, done, info

    def reset(self):
        self.close()
        self._launch_game()
        return self.observation

    def render(self, mode='human'):
        if mode == 'rgb_array':
            canvas = self._driver.find_element_by_id("mainCanvas")
            canvas.screenshot("screenshot.png")
            return imread("screenshot.png")
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            canvas = self._driver.find_element_by_id("mainCanvas")
            canvas.screenshot("screenshot.png")
            self.viewer.imshow(imread("screenshot.png", mode='RGB'))

            # WIP to avoid writing image on disk
            """array=np.fromstring(canvas.screenshot_as_png, dtype=np.uint8)

            from png import Reader
            png_reader=Reader(bytes=canvas.screenshot_as_png)
            width, height, pixels, meta= png_reader.read()
            pixels=np.array(list(map(np.uint16,pixels))).reshape((height,width,4))
            # Removing alpha
            mask = pixels[:,:,3]==255
            print(mask.shape)
            print(mask)
            print(np.where(mask,pixels,np.array([255,255,255,255],dtype=np.uint8).reshape((None,None,4))))
            self.viewer.imshow(pixels)"""
        return self.viewer.isopen

    def close(self):
        if self._driver is not None:
            self._driver.quit()
        if self.viewer is not None:
            self.viewer.close()

    @property
    def done(self):
        return self._is_dead

    ##########################
    #    Internal Methods
    ##########################

    def _launch_game(self):

        # Initialize parameters of a game
        self._my_block_id = None

        # Create a new instance of the Firefox driver
        binary = FirefoxBinary(self.firefox_path)
        firefox_options = Options()
        firefox_options.headless=True
        self._driver = webdriver.Firefox(firefox_binary=binary, firefox_options=firefox_options)

        # go to the splix.io home page
        self._driver.get("http://www.splix.io")
        play_u_i = self._driver.find_element_by_id("playUI")
        input_name = self._driver.find_element_by_id("nameInput")

        input_name.send_keys(self.__PLAYER_NAME)
        input_name.submit()

        element = WebDriverWait(self._driver, 10).until(
            expected_conditions.visibility_of(play_u_i)
        )

        # We stop the player
        dom = self._driver.find_element_by_tag_name("html")
        dom.send_keys("P")
        my_player = None
        while my_player is None:
            my_player = self._driver.execute_script("return myPlayer;")
        dom.send_keys("P")

        try:
            if self._driver.execute_script("return myPlayer;") is None:
                # We are on the landing page
                self._driver.close()
                self._launch_game()

        except:
            # the Web Driver has been closed, we have to launch it again
            self._launch_game()

        self._my_block_id = self.find_my_block_id()
        if self._my_block_id is None:
            self._driver.close()
            self._launch_game()

        # To hide Privacy Banner
        try:
            self._driver.execute_script(
                """document.getElementsByClassName("app_gdpr--2k2uB")[0].style.display="none" """)
        except:
            pass

    def _update_observations(self):
        """

        :return: None
        """

        script = f"""
        var res = new Object;
        res['my_direction']=myPlayer.dir;
        res['my_position']=myPlayer.pos.map(x=>Math.round(x));
        
        res['my_blocks']=blocks.filter(b=>b.currentBlock=={self._my_block_id}).map(b=>[b.x, b.y])
        res['free_blocks']=blocks.filter(b=>b.currentBlock==1).map(b=>[b.x, b.y])
        res['my_trails']=myPlayer.trails.reduce((res,item)=>res.concat(item.trail),[]);
        otherPlayers=players.filter(player =>player.id !=0)
        res['other_trails']=[];
        otherPlayers.forEach(function(player){{
          res['other_trails'].push(player.trails.reduce((res,item)=>res.concat(item.trail),[]))
        }})
        res['other_positions']=[];
        otherPlayers.forEach(function(player){{
          res['other_positions'].push(player.pos.map(x=>Math.round(x)))
        }})
        
        res['score']=myRealScoreElem.innerHTML;
        res['is_dead'] = myPlayer.isDead;
        res['kills']=myKillsElem.innerHTML;
        res['rank']=myRank;
        res['captured_blocks']=myScoreElem.innerHTML;
        return res;
        """

        info = dict(self._driver.execute_script(script))

        self._my_blocks = info['my_blocks']
        self._my_direction = info['my_direction']
        self._my_position = info['my_position']
        self._my_trails = info['my_trails']
        self._is_dead = info['is_dead']
        self._score = int(info['score'])
        self._free_blocks = info['free_blocks']
        self._captured_blocks = info['captured_blocks']
        self._rank = info['rank']
        self._kills = info['kills']

        self._other_trails = info['other_trails']
        self._other_positions = info['other_positions']

    @property
    def observation(self):
        self._update_observations()
        self._process_observation()
        return self._observation

    def find_my_block_id(self):
        script = """
        var potential_block_id = new Set()
        var counter = blocks.reduce((tally, block) => {
          tally[block.currentBlock] = (tally[block.currentBlock] || 0) + 1;
          return tally;
        }, {
        })
        for (var block_id in counter) {
          if (counter[block_id] == 25) {
            potential_block_id.add(block_id)
          }
        }
        if (potential_block_id.size == 1) {
          return potential_block_id.values().next().value
        }
        var myTrails = myPlayer.trails[0].trail
        if (myTrails.length > 0) {
          var myFirstTrail = myTrails[0]
          var x_FirstTrail = myFirstTrail[0]
          var y_FirstTrail = myFirstTrail[1]
          var potential_blocks = blocks.filter(b => potential_block_id.has(b.currentBlock.toString()))
          var close_blocks = potential_blocks.filter(b => (Math.abs(b.x - x_FirstTrail) + Math.abs(b.y - y_FirstTrail)) == 1)
          for (var b in close_blocks) {
            potential_block_id.add(b.currentBlock)
          }  // Removing block of first trail point
        
          forbiddenBlock = blocks.filter(b => (b.x == x_FirstTrail && b.y == y_FirstTrail))
          if (forbiddenBlock.length == 1) {
            potential_block_id.delete(forbiddenBlock[0].currentBlock)
          }
          if (potential_block_id.size == 1) {
            return potential_block_id.values().next().value
          } 
          else {
            return null
          }
        } 
        else {
            if (blocks.filter(b => (b.x == myPlayer.pos[0] && b.y == myPlayer.pos[1])).length ==1 ){
          return blocks.filter(b => (b.x == myPlayer.pos[0] && b.y == myPlayer.pos[1]))[0].currentBlock 
          }
          else{
          return null
          }
        }
        """

        return self._driver.execute_script(script)

    def _process_observation(self):

        state = np.zeros([self.__VIEW_SIZE, self.__VIEW_SIZE, 6], dtype=np.bool)

        if not self._is_dead:
            x_pos, y_pos = self._my_position
            x_min, x_max = int(x_pos - self.__VIEW_SIZE / 2), int(x_pos + self.__VIEW_SIZE / 2)
            y_min, y_max = int(y_pos - self.__VIEW_SIZE / 2), int(y_pos + self.__VIEW_SIZE / 2)

            # myBlocks
            for (x, y) in self._my_blocks:
                if x_min <= x < x_max and y_min <= y < y_max:
                    state[y - y_min, x - x_min, self.__FEATURES_CHANNELS['myBlocks']] = 1

            # (x,y) combinaison currentBlock==1
            for (x, y) in self._free_blocks:
                if x_min <= x < x_max and y_min <= y < y_max:
                    state[y - y_min, x - x_min, self.__FEATURES_CHANNELS['freeSpace']] = 1

            # From my Trail
            my_trail_points = self.points_from_trails(self._my_trails)
            for (x, y) in my_trail_points:
                if x_min <= x < x_max and y_min <= y < y_max:
                    state[y - y_min, x - x_min, self.__FEATURES_CHANNELS['myTrail']] = 1

            # For Other Trail

            for player_trails in self._other_trails:
                for (x, y) in self.points_from_trails(player_trails):
                    if x_min <= x < x_max and y_min <= y < y_max:
                        state[y - y_min, x - x_min, self.__FEATURES_CHANNELS['otherTrail']] = 1

            # For Other Position
            for (x, y) in self._other_positions:
                if x_min <= x < x_max and y_min <= y < y_max:
                    state[y - y_min, x - x_min, self.__FEATURES_CHANNELS['otherPositions']] = 1

            # Outside
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    if x < 0 or x >= self.__MAP_SIZE or y < 0 or y >= self.__MAP_SIZE:
                        state[y - y_min, x - x_min, self.__FEATURES_CHANNELS['outside']] = 1

            if self._my_direction in [1, 2, 3]:
                state = np.rot90(state, self._my_direction)

        self._observation = state

    @staticmethod
    def distance(x1, x2):
        return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

    @staticmethod
    def points_from_trails(trail):
        points = set()
        if trail:
            cur_x, cur_y = tuple(map(round, trail[0]))
            points.add((cur_x, cur_y))
            if len(trail) > 1:
                for next_point in trail[1:]:
                    nxt_x, nxt_y = tuple(map(round, next_point))
                    if nxt_x > cur_x:
                        points |= set(product(range(cur_x, nxt_x), [cur_y]))
                    elif nxt_x < cur_x:
                        points |= set(product(range(cur_x, nxt_x, -1), [cur_y]))
                    elif nxt_y > cur_y:
                        points |= set(product([cur_x], range(cur_y, nxt_y)))
                    else:
                        points |= set(product([cur_x], range(cur_y, nxt_y, -1)))
                    cur_x, cur_y = nxt_x, nxt_y
            points.add((cur_x, cur_y))

        return points

    def plot_screenshot(self):
        canvas = self._driver.find_element_by_id("mainCanvas")
        canvas.screenshot("screenshot.png")

    def plot_state(self):

        maps = np.split(self._observation, len(self.__FEATURES_CHANNELS), axis=2)

        for name, index in self.__FEATURES_CHANNELS.items():
            plt.title(name)
            if self._my_direction in [1, 2, 3]:
                plt.imshow(np.rot90(maps[index].reshape(self.__VIEW_SIZE, self.__VIEW_SIZE), -self._my_direction))
            else:
                plt.imshow(maps[index].reshape(self.__VIEW_SIZE, self.__VIEW_SIZE))

            plt.savefig("Screenshot/state_" + str(name) + ".png")


if __name__ == '__main__':
    env = SplixOnlineEnv()
    for _ in range(1):
        env.reset()
        while not env.done:
            env.render()
            env.step(env.action_space.sample())  # random action
        env.close()

