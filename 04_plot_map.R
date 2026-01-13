# Script that plots the map of Fig 1, along with Fig1a and Fig 1b #############

library(sf)
library(spdep)
library(ggspatial)
library(magick)
library(cowplot)
library(geobr)
library(ggthemes)
library(tidyverse)
f_img <- \(imgpath){
    vimg <- image_read(imgpath) %>%
        image_trim()
    image_write(vimg,imgpath)
}
#
## risco de reintrodução
df_score <- read_csv("data/risk_score.csv") %>%
    select(-1,-2)
## population estimates
load("data/pop_sarampo_novo.RData")
ano_ref <- 2024
df_pop <- pop_aux
rm(pop_aux)
df_pop <- df_pop %>%
  filter(Ano == ano_ref) %>%
  select(ID_MUN, POP) %>%
  group_by(ID_MUN) %>%
  summarise(POP = sum(POP))
df_pop <- df_pop %>% mutate(ID_MUN = as.integer(ID_MUN))
## shapefile dos municípios
map_sp <- read_municipality(code_muni = "SP",year = 2024)
map_sp <- map_sp %>% mutate(ID_MUN = substr(code_muni, 1, 6) %>% as.numeric) %>%
    rename("nm_mun" = 'name_muni') %>%
    select(ID_MUN,nm_mun)
## df_plot
df_plot <- left_join(map_sp,df_score) %>% left_join(.,df_pop)
#
#-----------------------
# estratégia do zoom
#-----------------------
# 1) filtrar os municípios
nb <- spdep::poly2nb(df_plot, queen=TRUE)
vi <- c(
    unlist(nb[df_plot$risk_category==1]),
    which(df_plot$risk_category==1)
    )
# 2) criar a caixa que inclui todos
bbox_zoom <- st_bbox(df_plot[vi,])
vajust <- -0.2
fator_expansao <- list(
    xmin = vajust,   # reduz à esquerda
    xmax = vajust,  # Reduz à direita
    ymin = vajust,   # reduz abaixo
    ymax = -vajust   # Reduz acima
)
bbox_ajustado <- c(
    bbox_zoom[1] + fator_expansao$xmin * (bbox_zoom[3] - bbox_zoom[1]),
    bbox_zoom[2] + fator_expansao$ymin * (bbox_zoom[4] - bbox_zoom[2]),
    bbox_zoom[3] + fator_expansao$xmax * (bbox_zoom[3] - bbox_zoom[1]),
    bbox_zoom[4] + fator_expansao$ymax * (bbox_zoom[4] - bbox_zoom[2])
)
class(bbox_ajustado) <- "bbox"  # Definir a classe correta
names(bbox_ajustado) <- c("xmin", "ymin", "xmax", "ymax")
# 3) selecionar todos os municípios dentro do box
df_zoom <- df_plot %>%
    filter(st_intersects(geom, st_as_sfc(bbox_zoom), sparse = FALSE)[,1])
nb2 <- poly2nb(df_zoom,queen = TRUE)
df_zoom <- df_zoom[which(sapply(nb2,length)!=1),] %>%
    filter(ID_MUN!=352030) %>% # remover municípios manualmente
    mutate(risk_cat = ifelse(risk_category==1,"high risk","low risk"),
           label=paste0("City: ",nm_mun,"\n",
                        "Pop: ",round(POP/1000,2)," k inhab.\n",
                        "Risk: ",round(prob,3)))
# 4) caixa usada para conectar os dois gráficos
bbox_zoom_sf <- st_as_sfc(bbox_ajustado)
st_crs(bbox_zoom_sf) <- st_crs(df_plot)
bbox_zoom_sf <- st_as_sfc(bbox_zoom)
if(FALSE){
    lp <- list()
    lp[[1]] <- df_zoom %>%
        ggplot(aes(fill=risk_category)) +
        geom_sf()
    lp[[2]] <-
        df_plot %>%
        # df_plot[vi,] %>%
        ggplot(aes(fill=risk_category)) +
        geom_sf() +
        geom_sf(data = bbox_zoom_sf, fill = NA, color = "red", linewidth = 1)
    gridExtra::grid.arrange(grobs=lp,ncol=2)
}
# 5) preparação para inserir as tabelas
library(ggrepel)
#
#
#-------------------------------
# lista de gráficos e elementos
#-------------------------------
#
#
f_ggplot1 <- \(size_scale=0.2,
               linew=0.25,
               legw=2,
               legh=0.5,
               leg_textsize=7.5,
               leg_titlesize=8,
               legsize=7.5,
               xscale=0.8,
               yscale=0.325,
               xarrow=0.75,
               yarrow=0.5,
               fillpos=c(0.1,0.2)){
    ggplot(df_plot,aes(fill=log10(prob))) +
        geom_sf(color="black",linewidth=0.05) +
        geom_sf(data = bbox_zoom_sf, fill = NA, color = "black", linewidth = linew) +
        scale_fill_distiller(palette = "Spectral", direction = -1) +
        labs(fill = 'Risk score (Log10)',
             title="a)") +
        annotation_scale(location = "br",
                         width_hint = 0.1,
                         pad_x = unit(xscale, "npc"),
                         pad_y = unit(yscale, "npc")) +
        annotation_north_arrow(location = "br",
                               which_north = "true",
                               pad_x = unit(xarrow, "npc"),
                               pad_y = unit(yarrow, "npc"),
                               style = north_arrow_fancy_orienteering) +
        coord_sf(expand = FALSE) +
        theme_bw() +
        theme(
            legend.title = element_text(size = leg_titlesize,
                                        face="bold",
                                        margin = margin(b = 10, r = 10)),
            legend.text  = element_text(size = leg_textsize),
            legend.position = fillpos,
            legend.direction = "horizontal",
            legend.box = "horizontal",
            legend.box.just = "center",
            legend.justification = "center",
            legend.title.position = "top",
            legend.key.width = unit(legw, "cm"),
            legend.key.height = unit(legh, "cm")
        )
}
f_ggplot2 <- \(slab=5,
               msl=2,
               bpad=2,
               legpos=c(0.2,0.5),
               df_nug,
               legsize=5,
               vpal=c("#ffd7b5","#ba93b2")){
    vpal <- vpal
    names(vpal) <- unique(df_zoom$risk_cat)
    dflab <- df_plot %>%
        filter(risk_category==1) %>%
        mutate(label=paste0("City: ",nm_mun,"\n",
                            "Pop: ",round(POP/1000,2)," k inhab.\n",
                            "Risk: ",round(prob,3))) %>%
        left_join(df_nug)
    df_plot %>%
        mutate(risk_cat = ifelse(risk_category==1,"high risk","low risk")) %>%
        ggplot(aes(fill=risk_cat)) +
        geom_sf(color="black",linewidth=0.05) +
        geom_label_repel(
            inherit.aes = FALSE,
            data=dflab,
            aes(label=label,geometry=geom),
            nudge_x = dflab$nx,
            nudge_y = dflab$ny,
            stat="sf_coordinates",
            size=slab,
            min.segment.length = msl,
            box.padding = bpad,
            segment.color = "black"
        ) +
        coord_sf(
            xlim = c(bbox_zoom["xmin"], bbox_zoom["xmax"]),
            ylim = c(bbox_zoom["ymin"], bbox_zoom["ymax"]),
            expand = FALSE
        ) +
        theme_bw() +
        labs(x="",y="",title="b)") +
        scale_fill_manual("Classification (threshold = 0.081)",values=vpal) +
        theme(legend.position=legpos,
              legend.text = element_text(size=legsize),
              legend.title = element_text(face="bold",size = 5),
              legend.title.position = "top",
              legend.direction = "horizontal",
              legend.box = "horizontal",
              legend.box.just = "center",
              legend.justification = "center",
              legend.key.size = unit(0.5, "cm"),
              legend.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "mm")
              )
}
lp <- list()
lp[[1]] <- f_ggplot1(size_scale=0.1,
                     legw=1,
                     legh=0.2,
                     linew=0.25,
                     fillpos=c(0.2,0.1),
                     xscale=0,
                     xarrow=0,
                     yscale=0.01,
                     yarrow=0.015)
dfnug <- data.frame(
    nm_mun = c("Americana","Santos","São Caetano do Sul","São Paulo"),
    nx = c(0.35,0.25,0.75,-0.75),
    ny = c(0,-0.275,0.25,0.25)
)
lp[[2]] <- f_ggplot2(legpos=c(0.8,0.05),
                     df_nug = dfnug,
                     slab = 2.5)
p <- plot_grid(plotlist = lp,ncol=2,align = "h",axis="tb",
          rel_widths = c(1, 0.58),
          labels = c("a)", "b)"),
          label_x = 0.05,
          label_y = 0.875)
save_plot(filename = "figs/fig1.jpeg",
          plot=p,
          base_width = 12,
          base_height = 8,
          dpi=300)
# segunda tentativa
names(lp) <- c("fig1a","fig1b")
f_ggsave <- \(li){
    p <- lp[[li]]
    ggsave(filename = paste0("figs/",li,".jpeg"),
           plot=p,
           dpi=200,
           height = 8,
           width = ifelse(li=="fig1a",12,12*0.58))
}
lapply(names(lp),f_ggsave)
lpath <- paste0("figs/",names(lp),".jpeg")
limg <- lapply(lpath, image_read) %>%
    lapply(.,image_trim)
img_final <- image_append(
        do.call("c",limg),
        stack = FALSE
    )
image_write(image = img_final,path = "figs/fig1.jpeg")