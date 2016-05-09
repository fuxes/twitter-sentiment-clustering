(function() {
  'use strict';

  angular
    .module('bubbleApp.models')
    .factory('Bubbles', Bubbles);

  Bubbles.$inject = [
    '$window'
  ];

  function Bubbles($window) {
    var bubbleClickHandlers = [];

    // Register global function for listening D3 clicks
    $window.bubbleClickEvent = bubbleClickEvent;

    return {
      onBubbleClick: onBubbleClick
    };

    function onBubbleClick(handler) {
      if (bubbleClickHandlers.indexOf(handler) < 0) {
        bubbleClickHandlers.push(handler);
      }
    }

    function bubbleClickEvent(target) {
      var len = bubbleClickHandlers.length;
      for (var i = 0; i < len; i++) {
        bubbleClickHandlers[i](target);
      }
    }
  }
})();