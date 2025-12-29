import pygame
import math
from constants import MAX_HP, MAX_AMMO, SHIELD_COOLDOWN_STEPS, RELOAD_STEPS


class PygameRenderer:
    """
    Enhanced pygame-based renderer for MicroTankArenaEnv.
    Features modern visuals with tank graphics, particle effects, and a polished HUD.
    """

    HUD_HEIGHT = 100

    # Color palette - Military/Neon hybrid theme
    COLORS = {
        # Background and floor
        'bg_dark': (12, 14, 18),
        'floor_base': (22, 26, 32),
        'floor_line': (35, 40, 50),
        'floor_glow': (40, 50, 65),
        
        # Walls
        'wall_top': (70, 75, 85),
        'wall_side': (45, 50, 60),
        'wall_shadow': (25, 28, 35),
        
        # Player tank (cyan/blue theme)
        'player_body': (30, 140, 180),
        'player_body_dark': (20, 100, 140),
        'player_turret': (50, 180, 220),
        'player_accent': (100, 220, 255),
        'player_glow': (0, 200, 255),
        
        # Enemy tank (red/orange theme)
        'enemy_body': (180, 60, 50),
        'enemy_body_dark': (140, 40, 35),
        'enemy_turret': (220, 80, 60),
        'enemy_accent': (255, 120, 80),
        'enemy_glow': (255, 80, 60),
        
        # Bullets - distinct colors matching tank themes
        'bullet_player': (100, 230, 255),  # Cyan to match player tank
        'bullet_enemy': (255, 120, 60),    # Orange to match enemy tank
        
        # Shield
        'shield_player': (0, 220, 255),
        'shield_enemy': (255, 100, 200),
        
        # HUD
        'hud_bg': (15, 18, 25),
        'hud_border': (50, 60, 80),
        'text_bright': (230, 235, 245),
        'text_dim': (140, 150, 170),
        'hp_player': (80, 200, 255),
        'hp_enemy': (255, 100, 100),
        'hp_bg': (40, 45, 55),
        'ammo_full': (255, 220, 80),
        'ammo_empty': (60, 55, 50),
        'cooldown_active': (255, 160, 60),
        'shield_ready': (100, 255, 200),
        'shield_cooldown': (80, 90, 110),
    }

    def __init__(self, grid_size, cell_size=40, fps=10):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self._closed = False
        self._frame_count = 0

        pygame.init()
        pygame.display.set_caption("⚔ MicroTank Arena ⚔")
        
        width = grid_size * cell_size
        height = grid_size * cell_size + self.HUD_HEIGHT
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Load fonts
        self._init_fonts()

    def _init_fonts(self):
        # Try to use nicer fonts, fall back to system fonts
        try:
            self.font_title = pygame.font.SysFont('Menlo', 16, bold=True)
            self.font_main = pygame.font.SysFont('Menlo', 14)
            self.font_small = pygame.font.SysFont('Menlo', 11)
            self.font_tiny = pygame.font.SysFont('Menlo', 10)
        except:
            self.font_title = pygame.font.SysFont(None, 20, bold=True)
            self.font_main = pygame.font.SysFont(None, 18)
            self.font_small = pygame.font.SysFont(None, 14)
            self.font_tiny = pygame.font.SysFont(None, 12)

    def render(self, env, hud_info=None):
        """Draw a frame of the environment plus an optional HUD."""
        if self._closed:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True
                pygame.quit()
                return False

        self._frame_count += 1
        
        # Fill background
        self.screen.fill(self.COLORS['bg_dark'])
        
        # Draw layers
        self._draw_floor(env)
        self._draw_walls(env)
        self._draw_bullets(env)
        self._draw_tanks(env)
        self._draw_hud(env, hud_info or {})

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _draw_floor(self, env):
        """Draw the arena floor with subtle grid pattern."""
        size = self.cell_size
        grid = env.current_map
        
        for y in range(env.grid_size):
            for x in range(env.grid_size):
                if grid[y, x] == 0:  # Floor tile
                    rect = pygame.Rect(x * size, y * size, size, size)
                    
                    # Base floor color with slight variation
                    base = self.COLORS['floor_base']
                    variation = ((x + y) % 2) * 3
                    floor_color = (base[0] + variation, base[1] + variation, base[2] + variation)
                    pygame.draw.rect(self.screen, floor_color, rect)
                    
                    # Subtle grid lines
                    pygame.draw.line(self.screen, self.COLORS['floor_line'], 
                                   (x * size, y * size), (x * size + size, y * size), 1)
                    pygame.draw.line(self.screen, self.COLORS['floor_line'],
                                   (x * size, y * size), (x * size, y * size + size), 1)
                    
                    # Corner accent dots
                    dot_color = self.COLORS['floor_glow']
                    pygame.draw.circle(self.screen, dot_color, (x * size + 2, y * size + 2), 1)

    def _draw_walls(self, env):
        """Draw 3D-ish walls with top and shadow."""
        size = self.cell_size
        grid = env.current_map
        depth = 4  # 3D depth effect
        
        for y in range(env.grid_size):
            for x in range(env.grid_size):
                if grid[y, x] == 1:  # Wall
                    # Shadow
                    shadow_rect = pygame.Rect(x * size + depth, y * size + depth, size, size)
                    pygame.draw.rect(self.screen, self.COLORS['wall_shadow'], shadow_rect)
                    
                    # Wall side (3D effect)
                    side_rect = pygame.Rect(x * size, y * size + size - depth, size, depth)
                    pygame.draw.rect(self.screen, self.COLORS['wall_side'], side_rect)
                    
                    # Wall top
                    top_rect = pygame.Rect(x * size, y * size, size, size - depth)
                    pygame.draw.rect(self.screen, self.COLORS['wall_top'], top_rect)
                    
                    # Border
                    pygame.draw.rect(self.screen, self.COLORS['wall_shadow'], 
                                   (x * size, y * size, size, size), 1)
                    
                    # Highlight on top-left edges
                    highlight = (self.COLORS['wall_top'][0] + 20, 
                               self.COLORS['wall_top'][1] + 20,
                               self.COLORS['wall_top'][2] + 20)
                    pygame.draw.line(self.screen, highlight,
                                   (x * size + 1, y * size + 1),
                                   (x * size + size - 2, y * size + 1), 1)
                    pygame.draw.line(self.screen, highlight,
                                   (x * size + 1, y * size + 1),
                                   (x * size + 1, y * size + size - depth - 1), 1)

    def _draw_tanks(self, env):
        """Draw tanks with proper body, turret, and effects."""
        state = env.state
        me = state["me"]
        enemy = state["enemy"]
        
        # Draw tanks with proper layering
        self._draw_single_tank(me['x'], me['y'], me['dir'], 'player',
                              me.get('shield_steps', 0) > 0)
        self._draw_single_tank(enemy['x'], enemy['y'], enemy['dir'], 'enemy',
                              enemy.get('shield_steps', 0) > 0)

    def _draw_single_tank(self, grid_x, grid_y, direction, tank_type, has_shield):
        """Draw a single tank with body, tracks, and turret."""
        size = self.cell_size
        cx = grid_x * size + size // 2
        cy = grid_y * size + size // 2
        
        # Color scheme based on tank type
        if tank_type == 'player':
            body_color = self.COLORS['player_body']
            body_dark = self.COLORS['player_body_dark']
            turret_color = self.COLORS['player_turret']
            accent = self.COLORS['player_accent']
            glow_color = self.COLORS['player_glow']
            shield_color = self.COLORS['shield_player']
        else:
            body_color = self.COLORS['enemy_body']
            body_dark = self.COLORS['enemy_body_dark']
            turret_color = self.COLORS['enemy_turret']
            accent = self.COLORS['enemy_accent']
            glow_color = self.COLORS['enemy_glow']
            shield_color = self.COLORS['shield_enemy']
        
        # Tank dimensions
        body_w = int(size * 0.7)
        body_h = int(size * 0.55)
        track_w = int(size * 0.15)
        turret_size = int(size * 0.35)
        barrel_len = int(size * 0.4)
        barrel_w = int(size * 0.12)
        
        # Calculate rotation angle
        angle = -direction * 90  # 0=up, 1=right, 2=down, 3=left
        
        # Draw tank shadow
        shadow_offset = 3
        self._draw_tank_body(cx + shadow_offset, cy + shadow_offset, 
                            body_w, body_h, track_w, angle,
                            (20, 22, 28), (20, 22, 28))
        
        # Draw tank body
        self._draw_tank_body(cx, cy, body_w, body_h, track_w, angle,
                            body_color, body_dark)
        
        # Draw turret
        pygame.draw.circle(self.screen, body_dark, (cx, cy), turret_size // 2 + 2)
        pygame.draw.circle(self.screen, turret_color, (cx, cy), turret_size // 2)
        
        # Draw barrel
        dx = [0, 1, 0, -1][direction]
        dy = [-1, 0, 1, 0][direction]
        barrel_end_x = cx + dx * barrel_len
        barrel_end_y = cy + dy * barrel_len
        
        # Barrel with outline
        pygame.draw.line(self.screen, body_dark, (cx, cy), 
                        (barrel_end_x, barrel_end_y), barrel_w + 2)
        pygame.draw.line(self.screen, turret_color, (cx, cy),
                        (barrel_end_x, barrel_end_y), barrel_w)
        
        # Barrel tip accent
        pygame.draw.circle(self.screen, accent, (int(barrel_end_x), int(barrel_end_y)), barrel_w // 2)
        
        # Shield effect
        if has_shield:
            self._draw_shield(cx, cy, size // 2 + 4, shield_color)
        
        # Subtle glow under tank
        glow_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        for r in range(3):
            alpha = 30 - r * 10
            pygame.draw.circle(glow_surface, (*glow_color[:3], alpha),
                             (size // 2, size // 2), size // 3 + r * 2)
        self.screen.blit(glow_surface, (grid_x * size, grid_y * size))

    def _draw_tank_body(self, cx, cy, body_w, body_h, track_w, angle, body_color, track_color):
        """Draw the tank body with tracks."""
        # Create surface for rotation
        surf_size = max(body_w, body_h) + track_w * 2 + 4
        tank_surface = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        center = surf_size // 2
        
        # Draw tracks (left and right)
        track_rect_l = pygame.Rect(center - body_w // 2 - track_w,
                                   center - body_h // 2,
                                   track_w, body_h)
        track_rect_r = pygame.Rect(center + body_w // 2,
                                   center - body_h // 2,
                                   track_w, body_h)
        pygame.draw.rect(tank_surface, track_color, track_rect_l, border_radius=2)
        pygame.draw.rect(tank_surface, track_color, track_rect_r, border_radius=2)
        
        # Track detail lines
        for i in range(4):
            y_pos = center - body_h // 2 + (body_h // 5) * (i + 1)
            pygame.draw.line(tank_surface, (track_color[0] - 20, track_color[1] - 20, track_color[2] - 20),
                           (center - body_w // 2 - track_w + 1, y_pos),
                           (center - body_w // 2 - 1, y_pos), 1)
            pygame.draw.line(tank_surface, (track_color[0] - 20, track_color[1] - 20, track_color[2] - 20),
                           (center + body_w // 2 + 1, y_pos),
                           (center + body_w // 2 + track_w - 1, y_pos), 1)
        
        # Draw main body
        body_rect = pygame.Rect(center - body_w // 2, center - body_h // 2, body_w, body_h)
        pygame.draw.rect(tank_surface, body_color, body_rect, border_radius=3)
        
        # Body detail - front indicator
        front_indicator = pygame.Rect(center - body_w // 4, center - body_h // 2,
                                      body_w // 2, 3)
        pygame.draw.rect(tank_surface, (body_color[0] + 30, body_color[1] + 30, body_color[2] + 30),
                        front_indicator)
        
        # Rotate and blit
        rotated = pygame.transform.rotate(tank_surface, angle)
        rot_rect = rotated.get_rect(center=(cx, cy))
        self.screen.blit(rotated, rot_rect)

    def _draw_shield(self, cx, cy, radius, color):
        """Draw animated shield effect."""
        pulse = math.sin(self._frame_count * 0.3) * 0.2 + 0.8
        
        # Outer glow
        for i in range(3):
            alpha = int(40 * pulse) - i * 10
            if alpha > 0:
                glow_surf = pygame.Surface((radius * 2 + 20, radius * 2 + 20), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*color[:3], alpha),
                                 (radius + 10, radius + 10), radius + 5 - i)
                self.screen.blit(glow_surf, (cx - radius - 10, cy - radius - 10))
        
        # Main shield ring
        pygame.draw.circle(self.screen, color, (cx, cy), int(radius * pulse), 3)
        
        # Sparkle points
        for i in range(6):
            angle = (self._frame_count * 5 + i * 60) * math.pi / 180
            sx = cx + math.cos(angle) * radius * pulse
            sy = cy + math.sin(angle) * radius * pulse
            pygame.draw.circle(self.screen, (255, 255, 255), (int(sx), int(sy)), 2)

    def _draw_bullets(self, env):
        """Draw simple colored bullets - cyan for player, orange for enemy."""
        size = self.cell_size
        
        for b in env.state["bullets"]:
            bx, by = b["x"], b["y"]
            if 0 <= bx < env.grid_size and 0 <= by < env.grid_size:
                cx = bx * size + size // 2
                cy = by * size + size // 2
                
                # Distinct colors based on owner
                if b["owner"] == "me":
                    color = self.COLORS['bullet_player']  # Cyan/yellow
                else:
                    color = self.COLORS['bullet_enemy']   # Orange/red
                
                # Simple bullet circle with small highlight
                pygame.draw.circle(self.screen, color, (cx, cy), size // 5)
                pygame.draw.circle(self.screen, (255, 255, 255), (cx - 1, cy - 1), size // 10)

    def _draw_hud(self, env, hud_info):
        """Draw modern HUD with stats and info."""
        width = self.grid_size * self.cell_size
        hud_y = self.grid_size * self.cell_size
        
        # HUD background
        hud_surface = pygame.Surface((width, self.HUD_HEIGHT), pygame.SRCALPHA)
        hud_surface.fill((*self.COLORS['hud_bg'], 240))
        
        # Top border line
        pygame.draw.line(hud_surface, self.COLORS['hud_border'], (0, 0), (width, 0), 2)
        
        # Gradient accent line
        for i in range(width):
            progress = i / width
            if progress < 0.5:
                color = self._lerp_color(self.COLORS['player_glow'], self.COLORS['hud_border'], progress * 2)
            else:
                color = self._lerp_color(self.COLORS['hud_border'], self.COLORS['enemy_glow'], (progress - 0.5) * 2)
            pygame.draw.line(hud_surface, color, (i, 2), (i, 4), 1)
        
        me = env.state["me"]
        enemy = env.state["enemy"]
        
        # Left section - Player stats
        self._draw_tank_stats(hud_surface, 10, 12, me, "PLAYER", 'player')
        
        # Right section - Enemy stats
        self._draw_tank_stats(hud_surface, width - 200, 12, enemy, "ENEMY", 'enemy')
        
        # Center section - Game info
        center_x = width // 2
        mode = hud_info.get("mode", "")
        episode = hud_info.get("episode")
        step = hud_info.get("step")
        epsilon = hud_info.get("epsilon")
        ep_reward = hud_info.get("ep_reward")
        
        y_offset = 15
        
        if mode:
            mode_text = self.font_title.render(mode.upper(), True, self.COLORS['text_bright'])
            hud_surface.blit(mode_text, (center_x - mode_text.get_width() // 2, y_offset))
            y_offset += 20
        
        if episode is not None or step is not None:
            info_text = f"EP {episode if episode is not None else '-'} | STEP {step if step is not None else '-'}"
            info_surf = self.font_small.render(info_text, True, self.COLORS['text_dim'])
            hud_surface.blit(info_surf, (center_x - info_surf.get_width() // 2, y_offset))
            y_offset += 16
        
        if epsilon is not None:
            eps_text = f"ε: {epsilon:.3f}"
            eps_surf = self.font_small.render(eps_text, True, self.COLORS['text_dim'])
            hud_surface.blit(eps_surf, (center_x - eps_surf.get_width() // 2, y_offset))
            y_offset += 16
        
        if ep_reward is not None:
            reward_color = self.COLORS['shield_ready'] if ep_reward >= 0 else self.COLORS['hp_enemy']
            reward_text = f"REWARD: {ep_reward:+.1f}"
            reward_surf = self.font_small.render(reward_text, True, reward_color)
            hud_surface.blit(reward_surf, (center_x - reward_surf.get_width() // 2, y_offset))
        
        self.screen.blit(hud_surface, (0, hud_y))

    def _draw_tank_stats(self, surface, x, y, tank, label, tank_type):
        """Draw stats for a single tank."""
        if tank_type == 'player':
            accent = self.COLORS['player_accent']
            hp_color = self.COLORS['hp_player']
        else:
            accent = self.COLORS['enemy_accent']
            hp_color = self.COLORS['hp_enemy']
        
        # Label
        label_surf = self.font_title.render(label, True, accent)
        surface.blit(label_surf, (x, y))
        
        # HP bar
        bar_y = y + 20
        bar_width = 120
        bar_height = 10
        hp_ratio = max(0, tank['hp']) / MAX_HP
        
        # Background
        pygame.draw.rect(surface, self.COLORS['hp_bg'], (x, bar_y, bar_width, bar_height), border_radius=3)
        
        # HP fill
        if hp_ratio > 0:
            fill_width = int(bar_width * hp_ratio)
            pygame.draw.rect(surface, hp_color, (x, bar_y, fill_width, bar_height), border_radius=3)
        
        # HP text
        hp_text = self.font_tiny.render(f"{tank['hp']}/{MAX_HP}", True, self.COLORS['text_bright'])
        surface.blit(hp_text, (x + bar_width + 5, bar_y - 1))
        
        # Ammo indicator
        ammo_y = bar_y + 16
        ammo_count = tank.get('ammo', 0)
        reload_active = tank.get('reload', 0) > 0
        
        for i in range(MAX_AMMO):
            ammo_x = x + i * 12
            if i < ammo_count:
                color = self.COLORS['ammo_full']
            else:
                color = self.COLORS['ammo_empty']
            pygame.draw.rect(surface, color, (ammo_x, ammo_y, 8, 6), border_radius=1)
        
        if reload_active:
            reload_text = self.font_tiny.render(f"RELOAD {tank.get('reload', 0)}", True, self.COLORS['cooldown_active'])
            surface.blit(reload_text, (x + MAX_AMMO * 12 + 5, ammo_y - 1))
        
        # Shield status
        shield_y = ammo_y + 14
        shield_steps = tank.get('shield_steps', 0)
        shield_cd = tank.get('shield_cooldown', 0)
        
        if shield_steps > 0:
            shield_text = f"◆ SHIELD ACTIVE ({shield_steps})"
            shield_color = self.COLORS['shield_ready']
        elif shield_cd > 0:
            shield_text = f"◇ SHIELD CD: {shield_cd}"
            shield_color = self.COLORS['shield_cooldown']
        else:
            shield_text = "◆ SHIELD READY"
            shield_color = self.COLORS['shield_ready']
        
        shield_surf = self.font_tiny.render(shield_text, True, shield_color)
        surface.blit(shield_surf, (x, shield_y))
        
        # Gun cooldown indicator
        cooldown = tank.get('cooldown', 0)
        if cooldown > 0:
            cd_text = self.font_tiny.render(f"GUN CD: {cooldown}", True, self.COLORS['cooldown_active'])
            surface.blit(cd_text, (x + 100, shield_y))

    def _lerp_color(self, c1, c2, t):
        """Linear interpolation between two colors."""
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )

    def close(self):
        """Clean up pygame resources."""
        if not self._closed:
            self._closed = True
            pygame.quit()
