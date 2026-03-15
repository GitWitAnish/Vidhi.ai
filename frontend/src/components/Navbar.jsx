import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { PanelLeft, Share, MoreHorizontal, Layers, List } from "lucide-react";
import logo from "/logo.jpeg";
import "./Navbar.css";

function Navbar({
  sidebarOpen,
  onToggleSidebar,
  showSubcategories,
  onToggleSubcategories,
}) {
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <motion.nav
      className="navbar"
      initial={{ y: -56, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <div className="navbar-left">
        <motion.button
          className={`navbar-btn sidebar-toggle ${!sidebarOpen || isMobile ? "visible" : "hidden"}`}
          onClick={onToggleSidebar}
          aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <PanelLeft size={20} />
        </motion.button>
        <motion.img
          src={logo}
          alt="VIDHI.AI"
          className="navbar-logo"
          whileHover={{ rotate: 360 }}
          transition={{ duration: 0.5 }}
        />
        <span className="navbar-title">VIDHI.AI</span>
      </div>
      <div className="navbar-right">
        <motion.button
          className="navbar-btn toggle-mode-btn"
          onClick={onToggleSubcategories}
          aria-label="Toggle subcategories view"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          transition={{ duration: 0.2 }}
        >
          {showSubcategories ? (
            <>
              <Layers size={18} />
              <span className="btn-text">Show Categories</span>
            </>
          ) : (
            <>
              <List size={18} />
              <span className="btn-text">Show Subcategories</span>
            </>
          )}
        </motion.button>
        <button className="navbar-btn" aria-label="Share">
          <Share size={18} />
        </button>
      </div>
    </motion.nav>
  );
}

export default Navbar;
