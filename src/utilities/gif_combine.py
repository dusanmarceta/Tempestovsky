from PIL import Image

def append_gifs(path1, path2, output_path):
    gif1 = Image.open(path1)
    gif2 = Image.open(path2)
    
    frames = []

    # Izvuci sve frejmove iz prvog GIF-a
    for i in range(gif1.n_frames):
        gif1.seek(i)
        frames.append(gif1.copy())
    
    # Izvuci sve frejmove iz drugog GIF-a
    for i in range(gif2.n_frames):
        gif2.seek(i)
        frames.append(gif2.copy())

    # Sačuvaj kao jedan fajl
    # duration je u milisekundama (npr. 40ms = 25fps)
    frames[0].save(
        output_path, 
        save_all=True, 
        append_images=frames[1:], 
        loop=0, 
        duration=gif1.info.get('duration', 40)
    )

# Korišćenje:
append_gifs('tumbling_animation_1.gif', 'tumbling_animation_2.gif', 'tumbling_animation.gif')