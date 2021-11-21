

for event in pygame.event.get():
    if event.type == pygame.QUIT:
        running = False
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        x, y = event.pos()
        row = round((y - 40) / 40)
        col = round((x - 40) / 40)