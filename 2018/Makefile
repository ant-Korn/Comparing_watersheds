IMG_FOLDER=dest_imgs
SOURCE_FOLDER=source_imgs
SIZES_FILE=sizes.txt
LOGS_FOLDER=logs_watershed
PLOTS_FOLDER=plots

SOURCE_3D_F=source_imgs_3D

init: clean $(IMG_FOLDER) $(LOGS_FOLDER) $(PLOTS_FOLDER) gen_imgs gen_3D

compare_all_2D: compare_2D_time compare_2D_time_WL compare_2D_mem compare_2D_mem_WL

compare_all_3D: compare_3D_time compare_3D_time_WL compare_3D_mem compare_3D_mem_WL

$(SOURCE_3D_F):
	mkdir $(SOURCE_3D_F)

$(PLOTS_FOLDER):
	mkdir $(PLOTS_FOLDER)

$(IMG_FOLDER):
	mkdir $(IMG_FOLDER)

$(LOGS_FOLDER):
	mkdir -p $(LOGS_FOLDER)/{csv,3D/csv}

gen_imgs: clean_2D $(SIZES_FILE) resize_src_imgs.sh $(SOURCE_FOLDER) $(IMG_FOLDER)
	./resize_src_imgs.sh -sd $(SOURCE_FOLDER) -dd $(IMG_FOLDER) -f $(SIZES_FILE)

$(SIZES_FILE): generate_size.py 
	python generate_size.py -N 40 -st 400000 -init 512 > $(SIZES_FILE)

compare_2D_time: $(IMG_FOLDER) compare_2D.py
	python compare_2D.py -d dest_imgs --resource time --logs_dir $(LOGS_FOLDER)

compare_2D_time_WL: $(IMG_FOLDER) compare_2D.py
	python compare_2D.py -d dest_imgs --resource time --logs_dir $(LOGS_FOLDER) -wl

compare_2D_mem: $(IMG_FOLDER) compare_2D.py
	python compare_2D.py -d dest_imgs --resource memory --logs_dir $(LOGS_FOLDER)

compare_2D_mem_WL: $(IMG_FOLDER) compare_2D.py
	python compare_2D.py -d dest_imgs --resource memory --logs_dir $(LOGS_FOLDER) -wl

compare_3D_time: compare_3D.py $(SOURCE_3D_F)
	python compare_3D.py -d source_imgs_3D --resource time --logs_dir $(LOGS_FOLDER)

compare_3D_time_WL: $(IMG_FOLDER) compare_2D.py
	python compare_3D.py -d source_imgs_3D --resource time --logs_dir $(LOGS_FOLDER) -wl

compare_3D_mem: $(IMG_FOLDER) compare_2D.py
	python compare_3D.py -d source_imgs_3D --resource memory --logs_dir $(LOGS_FOLDER)

compare_3D_mem_WL: $(IMG_FOLDER) compare_2D.py
	python compare_3D.py -d source_imgs_3D --resource memory --logs_dir $(LOGS_FOLDER) -wl

plot_all: plots_time plots_mem

plots_time: plot_graph_time.py
	python plot_graph_time.py -d 2 -lg
	python plot_graph_time.py -d 2 -lg -wl
	python plot_graph_time.py -d 3 -lg
	python plot_graph_time.py -d 3 -lg -wl

plots_mem: plot_graph_mem.py
	python plot_graph_mem.py -d 2
	python plot_graph_mem.py -d 2 -wl
	python plot_graph_mem.py -d 3
	python plot_graph_mem.py -d 3 -wl

clean: clean_2D clean_3D

clean_2D: $(IMG_FOLDER)
	rm -f $(IMG_FOLDER)/*
	rm -f $(SIZES_FILE)

clean_3D: $(SOURCE_3D_F)
	rm -rf $(SOURCE_3D_F)/*

gen_3D: clean_3D gen_3D.py $(SOURCE_3D_F)
	python gen_3D.py $(SOURCE_3D_F)
